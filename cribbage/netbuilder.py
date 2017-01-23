#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
netbuilder.py
(c) Will Roberts  11 January, 2017

Declarative description language for producing persistent neural
network models.
'''

from __future__ import absolute_import
import itertools
import json
import logging
import os
import sys
import time
from cribbage.utils import grouped, mkdir_p, open_atomic
import lasagne
import numpy as np
import theano
import theano.tensor as T

# logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ModelStore(object):
    '''
    An object representing a directory where Model objects are saved to
    disk.
    '''

    def __init__(self, path):
        '''
        Constructor.

        Arguments:
        - `path`:
        '''
        self.path = path

    def ensure_exists(self):
        '''Ensure the directory used by this ModelStore exists.'''
        mkdir_p(self.abs_path)

    @property
    def abs_path(self):
        '''Get this ModelStore's absolute path.'''
        return os.path.abspath(self.path)

NONLINEARITY_NAMES = {
    'LeakyRectify': lasagne.nonlinearities.LeakyRectify,
    'ScaledTanH': lasagne.nonlinearities.ScaledTanH,
    'ScaledTanh': lasagne.nonlinearities.ScaledTanh,
    'identity': lasagne.nonlinearities.identity,
    'leaky_rectify': lasagne.nonlinearities.leaky_rectify,
    'linear': lasagne.nonlinearities.linear,
    'rectify': lasagne.nonlinearities.rectify,
    'sigmoid': lasagne.nonlinearities.sigmoid,
    'softmax': lasagne.nonlinearities.softmax,
    'tanh': lasagne.nonlinearities.tanh,
    'theano': lasagne.nonlinearities.theano,
    'very_leaky_rectify': lasagne.nonlinearities.very_leaky_rectify,
}

OBJECTIVE_NAMES = {
    'binary_crossentropy': lasagne.objectives.binary_crossentropy,
    'categorical_crossentropy': lasagne.objectives.categorical_crossentropy,
    'squared_error': lasagne.objectives.squared_error,
    'binary_hinge_loss': lasagne.objectives.binary_hinge_loss,
    'multiclass_hinge_loss': lasagne.objectives.multiclass_hinge_loss,
    }

UPDATE_NAMES = {
    'adadelta': lasagne.updates.adadelta,
    'adagrad': lasagne.updates.adagrad,
    'adam': lasagne.updates.adam,
    'momentum': lasagne.updates.momentum,
    'nesterov_momentum': lasagne.updates.nesterov_momentum,
    'rmsprop': lasagne.updates.rmsprop,
    'sgd': lasagne.updates.sgd,
}

class NetworkWrapper(object):
    '''An object which wraps a Lasagne feedforward neural network.'''

    def __init__(self):
        '''Constructor.'''
        self.objective_name = 'squared_error'
        self.update_name = 'adadelta'
        # update parameters, e.g., learning rate, for training
        self.update_args_value = {}
        # this variable holds the actual neural network
        self._network = None
        # functions which can scale the input and output of the network
        self.input_scaler_fn = None
        self.output_scaler_fn = None
        # pointers to compiled theano functions
        self.train_fn = None
        self.validation_fn = None
        self.output_fn = None

    def _build_network(self, network_arch):
        '''
        Builds the Lasagne network for this object from the description in
        `network_arch`.
        '''
        assert len(network_arch) > 1
        assert network_arch[0]['layer'] == 'input'
        assert network_arch[-1]['layer'] == 'output'
        if self._network is None:
            num_hidden_layers = 0
            for layer in network_arch:
                if layer['layer'] == 'input':
                    assert self._network is None
                    self._network = lasagne.layers.InputLayer(
                        shape=(None, layer['size']), name='input')
                else:
                    assert self._network is not None
                    nonlinearity = NONLINEARITY_NAMES[layer['activation']]
                    if layer['layer'] == 'hidden':
                        num_hidden_layers += 1
                        name = 'hidden{}'.format(num_hidden_layers)
                    else:
                        name = 'output'
                    self._network = lasagne.layers.DenseLayer(
                        self._network, num_units=layer['size'],
                        name=name,
                        nonlinearity=nonlinearity)
                if 'dropout' in layer and layer['dropout'] is not None:
                    self._network = lasagne.layers.DropoutLayer(self._network, p=layer['dropout'])

    @property
    def network_layers(self):
        '''
        Returns a dictionary mapping layer names to lasagne Layer objects.
        '''
        return dict((layer.name, layer) for layer in
                    lasagne.layers.get_all_layers(self._network)
                    if layer.name is not None)

    def load_params(self, filename):
        '''
        Loads network parameters (weight matrices) from the given numpy
        .npz file into this object's neural network.

        Arguments:
        - `filename`:
        '''
        with np.load(filename) as input_file:
            param_values = [input_file['arr_%d' % i] for i in
                            range(len(input_file.files))]
        lasagne.layers.set_all_param_values(self._network, param_values)

    def get_layer(self, layer_name):
        '''
        Returns the lasagne Layer from this objects's neural network with
        the given name.

        Arguments:
        - `layer_name`:
        '''
        return self.network_layers[layer_name]

    def set_weights(self, layer_name, values):
        '''
        Sets the weight parameters (weight matrix and bias) on the given
        layer of this network to the values given.

        Arguments:
        - `layer_name`:
        - `values`: a list of weight parameter matrices
        '''
        params = self.get_layer(layer_name).get_params()
        assert len(values) == len(params)
        for (value, param) in zip(values, params):
            assert value.shape == param.get_value().shape
            param.set_value(value)

    def get_weights(self, layer_name):
        '''
        Returns the weight parameters (weight matrix and bias) for the
        given layer of this network.

        Arguments:
        - `layer_name`:
        '''
        return [param.get_value() for param in self.get_layer(layer_name).get_params()]

    def objective(self, objective_fn):
        '''
        Sets the objective function used for training.  This defaults to
        'squared_error'.

        Arguments:
        - `objective_fn`:
        '''
        self.objective_name = objective_fn

    def update(self, update_fn):
        '''
        Sets the update method used for training.  This defaults to
        'adadelta'.

        Arguments:
        - `update_fn`:
        '''
        self.update_name = update_fn

    def update_args(self, params):
        '''
        Sets update params (e.g., the learning rate) to use for training.

        Arguments:
        - `params`: a dictionary with keywords and values
        '''
        self.update_args_value = params

    def input_scaler(self, scaler_fn):
        '''Sets a function which scales the network inputs.'''
        self.input_scaler_fn = scaler_fn

    def output_scaler(self, scaler_fn):
        '''Sets a function which scales the network outputs.'''
        self.output_scaler_fn = scaler_fn

    def get_theano_functions(self):
        '''
        Returns pointers to compiled theano functions using this Model's
        network.

        Returns a tuple (train_fn, validation_fn, output_fn).
        '''
        if self.train_fn is None:
            # theano variables for inputs and outputs
            inputs = T.matrix('inputs')
            outputs = T.matrix('outputs')

            # the raw output of the network
            predictions = lasagne.layers.get_output(self._network, inputs)

            # define the loss function between the network output and the
            # training output
            objective_fn = OBJECTIVE_NAMES[self.objective_name]
            # TODO: some objective functions can take optional
            # parameters (e.g., delta for binary_hinge_loss, etc.)
            # TODO: parameterise the aggregation method too?
            loss = objective_fn(predictions, outputs)
            loss = lasagne.objectives.aggregate(loss, mode='mean')

            # for validation, we use the network in deterministic mode (e.g.,
            # fix dropout)
            deterministic_predictions = lasagne.layers.get_output(
                self._network, inputs, deterministic=True)

            # validation loss uses the same objective as training loss
            validation_loss = objective_fn(deterministic_predictions, outputs)
            validation_loss = lasagne.objectives.aggregate(validation_loss, mode='mean')

            # retrieve all trainable parameters from the model's neural network
            params = lasagne.layers.get_all_params(self._network, trainable=True)

            # define the update function
            update_fn = UPDATE_NAMES[self.update_name]
            updates = update_fn(loss, params, **self.update_args_value)

            # calculate magnitudes of updates
            update_mags = [((param - updates[param]) ** 2).mean() for param in params]
            self.update_mags_fn = (lambda f: (lambda i, o: np.array(f(i,o))))(
                theano.function([inputs, outputs], update_mags))
            # calculate magnitudes of params
            param_mags = [(param ** 2).mean() for param in params]
            self.param_mags_fn = (lambda f: (lambda: np.array(f())))(
                theano.function([], param_mags))

            # compile the training and validation functions in theano
            self.train_fn = theano.function([inputs, outputs], loss, updates=updates)
            self.validation_fn = theano.function([inputs, outputs], validation_loss)
            self.output_fn = theano.function([inputs], deterministic_predictions)
        return self.train_fn, self.validation_fn, self.output_fn, self.update_mags_fn, self.param_mags_fn

    def compute(self, inputs):
        '''
        Runs the given inputs through the neural network and returns the
        output.

        Arguments:
        - `inputs`:
        '''
        _tf, _vf, output_fn, _umf, _pmf = self.get_theano_functions()
        if self.input_scaler_fn:
            inputs = self.input_scaler_fn(inputs)
        output = output_fn(inputs)
        if self.output_scaler_fn:
            output = self.output_scaler_fn(output)
        return output

class Model(NetworkWrapper):
    '''An object wrapping a Lasagne feedforward neural network.'''

    MAX_VALIDATION_SET_SIZE = 10000

    def __init__(self, store, model_name):
        '''
        Constructor

        Arguments:
        - `store`: a ModelStore or a string to initialise a ModelStore
          with
        - `model_name`:
        '''
        super(Model, self).__init__()
        if not isinstance(store, ModelStore):
            store = ModelStore(store)
        self.store = store
        self.model_name = model_name
        # validation is computed after this many minibatches have been
        # trained
        self.validation_interval = 50
        # this is a tuple of (np.array, np.array) representing inputs
        # and outputs; if it is not None, it is used between
        # minibatches to compute validation
        self.validation_set = None
        # if not None, this function is passed this object to compute
        # a validation statistic
        self.use_validation_routine = None
        # these are iterables of np.array values representing inputs
        # and outputs; if not None, they are used for training
        self.training_inputs = None
        self.training_outputs = None
        # this value is set to a positive integer (length of training
        # set) if the training set for this Model is of finite size;
        # it is set to False otherwise
        self.finite_training_set = False
        # minibatch size; if this is not None, training (input,
        # output) pairs are grouped into blocks of this size during
        # training
        self.minibatch_size_value = None
        # training length
        # number of minibatches to train; if this is not None,
        # training stops after this many minibatches have been seen
        self.use_num_minibatches = None
        # number of epochs to train; if this is not None, training
        # stops after this many loops through the training set.  only
        # use if training set is of finite size.
        self.use_num_epochs = None
        # metadata dictionary for this Model
        self.metadata = None
        # is the network architecture specified by the metadata file?
        self.arch_frozen = False
        # are the network's weights loaded from a snapshot?
        self.weights_loaded = False
        # if the architecture is loaded from metadata, the input(),
        # hidden() and output() methods are used to check; this
        # variable stores which layer we're currently checking
        self.arch_check_stage = 0
        # this flag indicates if the network architecture has been
        # fully specified (for new Models) or has been fully checked
        # (for Models loaded from disk)
        self.arch_desc_complete = False
        # load metadata if possible
        self.ensure_exists()
        try:
            self.load_metadata()
            if 'architecture' in self.metadata:
                self.arch_frozen = True
            if 'snapshots' in self.metadata and len(self.metadata['snapshots']) > 0:
                # load the last snapshot into the network
                last_snapshot_fn = self.metadata['snapshots'][-1]['filename']
                self.network # build network
                self.load_params(os.path.join(self.model_path, last_snapshot_fn))
                logger.info('weights loaded from snapshot %s', last_snapshot_fn)
                self.weights_loaded = True
        except IOError:
            # no metadata file present, initialise metadata
            self.metadata = {
                'num_training_samples': 0,
                'num_epochs': 0,
                'num_minibatches': 0,
                'snapshots': [],
            }
            self.save_metadata()

    @property
    def model_path(self):
        '''Get the path where this Model's files are stored.'''
        return os.path.join(self.store.abs_path, self.model_name)

    def ensure_exists(self):
        '''Ensure the directory used by this Model exists.'''
        mkdir_p(self.model_path)

    @property
    def metadata_filename(self):
        '''Get this Model's metadata filename.'''
        return os.path.join(self.model_path, 'metadata.json')

    def load_metadata(self):
        '''Loads metadata for this Model from disk.'''
        with open(self.metadata_filename, 'rb') as input_file:
            self.metadata = json.loads(input_file.read().decode('utf-8'))
        self.objective_name = self.metadata['objective_name']
        self.update_name = self.metadata['update_name']
        self.update_args_value = self.metadata['update_args_value']

    def save_metadata(self):
        '''Saves metadata for this Model to disk.'''
        self.metadata['objective_name'] = self.objective_name
        self.metadata['update_name'] = self.update_name
        self.metadata['update_args_value'] = self.update_args_value
        with open_atomic(self.metadata_filename, 'wb') as output_file:
            output_file.write(json.dumps(self.metadata, indent=4))

    @property
    def best_validation_error(self):
        '''
        Property accessor: shortcut for Model.load_snapshot('best_validation')
        '''
        return self.load_snapshot('best_validation')

    def input(self, input_size, dropout=None):
        '''
        Creates an input layer of the given size.

        Arguments:
        - `input_size`:
        '''
        assert not self.arch_desc_complete
        if self.arch_frozen:
            # we're checking the network architecture against the
            # metadata now
            assert 'architecture' in self.metadata
            assert len(self.metadata['architecture']) > 1
            input_layer = self.metadata['architecture'][0]
            assert input_layer['layer'] == 'input'
            assert input_layer['size'] == input_size
            assert input_layer['dropout'] == dropout
            self.arch_check_stage = 1
        else:
            # we're specifying the network architecture now
            assert 'architecture' not in self.metadata
            self.metadata['architecture'] = []
            self.metadata['architecture'].append({'layer': 'input',
                                                  'size': input_size,
                                                  'dropout': dropout})

    def hidden(self, hidden_size, activation='sigmoid', dropout=None):
        '''
        Creates a hidden layer of the given size.

        Arguments:
        - `hidden_size`:
        - `activation`:
        - `dropout`: if not None, the probability that a node's output
          will be dropped
        '''
        assert 'architecture' in self.metadata
        assert not self.arch_desc_complete
        if self.arch_frozen:
            # we're checking the network architecture against the
            # metadata now
            assert len(self.metadata['architecture']) > 1
            assert self.arch_check_stage > 0
            current_layer = self.metadata['architecture'][self.arch_check_stage]
            assert current_layer['layer'] == 'hidden'
            assert current_layer['size'] == hidden_size
            assert current_layer['activation'] == activation
            assert current_layer['dropout'] == dropout
            self.arch_check_stage += 1
        else:
            # we're specifying the network architecture now
            self.metadata['architecture'].append({'layer': 'hidden',
                                                  'size': hidden_size,
                                                  'activation': activation,
                                                  'dropout': dropout})

    def output(self, output_size, activation='sigmoid'):
        '''
        Creates an output layer of the given size.

        Arguments:
        - `output_size`:
        - `activation`:
        '''
        assert 'architecture' in self.metadata
        assert not self.arch_desc_complete
        if self.arch_frozen:
            # we're checking the network architecture against the
            # metadata now
            assert len(self.metadata['architecture']) > 1
            assert self.arch_check_stage > 0
            current_layer = self.metadata['architecture'][self.arch_check_stage]
            assert current_layer['layer'] == 'output'
            assert current_layer['size'] == output_size
            assert current_layer['activation'] == activation
            assert len(self.metadata['architecture']) == self.arch_check_stage + 1
        else:
            # we're specifying the network architecture now
            self.metadata['architecture'].append({'layer': 'output',
                                                  'size': output_size,
                                                  'activation': activation})
        self.arch_desc_complete = True

    @property
    def network(self):
        '''Returns this Model's neural network object.'''
        if not self._network:
            assert 'architecture' in self.metadata
            self._build_network(self.metadata['architecture'])
        return self._network

    @property
    def network_layers(self):
        '''
        Returns a dictionary mapping layer names to lasagne Layer objects.
        '''
        return dict((layer.name, layer) for layer in
                    lasagne.layers.get_all_layers(self.network)
                    if layer.name is not None)

    def set_weights(self, layer_name, values):
        '''
        Sets the weight parameters (weight matrix and bias) on the given
        layer of this network to the values given.

        Arguments:
        - `layer_name`:
        - `values`: a list of weight parameter matrices
        '''
        if self.weights_loaded:
            logger.info('weights loaded from snapshot; will not '
                        'overwrite weights for layer %s', layer_name)
        else:
            super(Model, self).set_weights(layer_name, values)

    def get_theano_functions(self):
        '''
        Returns pointers to compiled theano functions using this Model's
        network.

        Returns a tuple (train_fn, validation_fn, output_fn).
        '''
        _nw = self.network # force build network
        return super(Model, self).get_theano_functions()

    def save_snapshot(self, train_err, validation_err, elapsed_time):
        '''
        Saves a snapshot of this Model's network to disk.  Also records
        metadata about the snapshot, such as training and validation
        error.

        Arguments:
        - `train_err`:
        - `validation_err`:
        - `elapsed_time`: the amount of time since the last snapshot,
          in seconds
        '''
        snapshot_filename = os.path.join(
            self.model_path,
            '{:010d}.npz'.format(self.metadata['num_minibatches']))
        np.savez(snapshot_filename, *lasagne.layers.get_all_param_values(self.network))
        total_time = 0.
        if 'snapshots' in self.metadata and len(self.metadata['snapshots']) > 0:
            total_time = self.metadata['snapshots'][-1]['total_time']
        self.metadata['snapshots'].append({
            'num_minibatches': self.metadata['num_minibatches'],
            'train_err': train_err,
            'validation_err': validation_err,
            'total_time': total_time + elapsed_time,
            'filename': os.path.basename(snapshot_filename),
        })
        self.save_metadata()

    def load_snapshot(self, snapshot_id):
        '''
        Loads a previously saved snapshot of this Model and returns it.

        Arguments:
        - `snapshot_id`: snapshot to load; this can be specified as an
          integer (i.e., the value of 'num_minibatches'), or as a
          special string value (one of 'first', 'last',
          'best_validation', 'best_train')
        '''
        # locate the snapshot filename to load
        assert 'snapshots' in self.metadata and len(self.metadata['snapshots']) > 0
        snapshots = self.metadata['snapshots']
        if isinstance(snapshot_id, int):
            candidates = [ss['filename'] for ss in snapshots
                          if ss['num_minibatches'] == snapshot_id]
            if candidates:
                snapshot_filename = candidates[0]
            else:
                raise Exception('Could not find snapshot with snapshot ID {}'.format(snapshot_id))
        elif snapshot_id == 'first':
            snapshot_filename = snapshots[0]['filename']
        elif snapshot_id == 'last':
            snapshot_filename = snapshots[-1]['filename']
        elif snapshot_id == 'best_validation':
            snapshot_filename = min([(ss['validation_err'], ss['filename']) for ss in snapshots])[1]
        elif snapshot_id == 'best_train':
            snapshot_filename = min([(ss['train_err'], ss['filename']) for ss in snapshots])[1]

        assert 'architecture' in self.metadata
        snapshot = NetworkWrapper()
        snapshot._build_network(self.metadata['architecture'])
        # copy over this Model's inherited attributes
        snapshot.objective_name = self.objective_name
        snapshot.update_name = self.update_name
        snapshot.update_args_value = self.update_args_value
        snapshot.load_params(os.path.join(self.model_path, snapshot_filename))
        return snapshot

    def validation(self, validation_set):
        '''
        Sets the validation set used for validation during training.

        `validation_set` can be an iterable, but it must be of finite
        length.

        Arguments:
        - `validation_set`:
        '''
        # validation set might be a tuple: (inputs, outputs)
        if isinstance(validation_set, tuple) and len(validation_set) == 2:
            inputs, outputs = validation_set
            inputs = np.array(list(itertools.islice(
                inputs, self.MAX_VALIDATION_SET_SIZE)))
            outputs = np.array(list(itertools.islice(
                outputs, self.MAX_VALIDATION_SET_SIZE)))
            assert len(inputs) == len(outputs)
            self.validation_set = (inputs, outputs)
        else:
            # otherwise, it might be an iterable
            #
            # the iterable is taken to consist of tuples of (input,
            # output) pairs
            #
            # truncate validation set to maximum allowed size
            validation_set = list(itertools.islice(
                validation_set, self.MAX_VALIDATION_SET_SIZE))
            inputs = np.array([i for (i, o) in validation_set])
            outputs = np.array([o for (i, o) in validation_set])
            self.validation_set = (inputs, outputs)

    def validation_routine(self, validation_routine):
        '''
        Sets the validation function used for validation during training.
        The argument should be a function which, when passed this
        object, returns a validation statistic.

        Arguments:
        - `validation_routine`:
        '''
        self.use_validation_routine = validation_routine

    def training(self, training_set):
        '''
        Sets the training set used for training.

        Arguments:
        - `training_set`:
        '''
        # training set might be a tuple: (inputs, outputs)
        if isinstance(training_set, tuple) and len(training_set) == 2:
            self.training_inputs, self.training_outputs = training_set
            # try to detect if the training set is finite or not
            try:
                self.finite_training_set = len(self.training_inputs)
            except TypeError:
                self.finite_training_set = False
        else:
            # otherwise, it might be an iterable
            #
            # the iterable is taken to consist of tuples of (input,
            # output) pairs
            #
            # try to detect if the training set is finite or not
            try:
                self.finite_training_set = len(training_set)
            except TypeError:
                self.finite_training_set = False
            inputs, outputs = itertools.tee(training_set)
            self.training_inputs = (i for (i, o) in inputs)
            self.training_outputs = (o for (i, o) in outputs)

    def minibatch_size(self, minibatch_size):
        '''
        Sets the size of minibatches to use for training.

        Arguments:
        - `minibatch_size`:
        '''
        self.minibatch_size_value = minibatch_size

    def num_minibatches(self, num_minibatches):
        '''
        Configures how long training should run for.  Useful if the
        training set is infinitely long.

        Arguments:
        - `num_minibatches`:
        '''
        self.use_num_minibatches = num_minibatches

    def num_epochs(self, num_epochs):
        '''
        Configures how long training should run for, if the training set
        is of finite length.

        Arguments:
        - `num_epochs`:
        '''
        self.use_num_epochs = num_epochs

def minibatcher(num, iterable):
    '''
    Wraps a `grouped` generator around `iterable` and then returns the
    result inside a numpy array.

    Arguments:
    - `num`:
    - `iterable`:
    '''
    for item in grouped(num, iterable):
        yield np.array(item)

def build(model, max_num_epochs = None, max_num_minibatches = None):
    '''
    Builds a model and trains it up until its training criterion is
    satisfied.

    Arguments:
    - `model`:
    - `max_num_epochs`: optional, the maximum number of epochs to
      train
    - `max_num_minibatches`: optional, the maximum number of
      minibatches to train
    '''
    train_fn, validation_fn, _of, update_mag_fn, param_mag_fn = model.get_theano_functions()

    # handle minibatching if specified by the model
    if model.minibatch_size_value is not None:
        minibatcher_fn = lambda xs: minibatcher(model.minibatch_size_value, xs)
    else:
        minibatcher_fn = lambda xs: xs

    # handle input scaling
    if model.input_scaler_fn is not None:
        input_scaler_fn = model.input_scaler_fn
    else:
        input_scaler_fn = lambda xs: xs

    # TODO handle output scaling

    if max_num_minibatches is None and model.use_num_minibatches is not None:
        max_num_minibatches = model.use_num_minibatches

    # if training set is finite and max_num_epochs or num_epochs is
    # specified, we loop that many times; otherwise, we loop once
    counting_epochs = False
    if model.finite_training_set:
        if max_num_epochs is not None:
            counting_epochs = True
        elif model.use_num_epochs is not None:
            max_num_epochs = model.use_num_epochs
            counting_epochs = True
        else:
            max_num_epochs = 1
    else:
        max_num_epochs = 1

    for epoch in range(max_num_epochs):
        # training loop
        start_time = time.time()
        train_err = 0
        for num_minibatches, (input_minibatch, output_minibatch) in enumerate(
                itertools.izip(minibatcher_fn(model.training_inputs),
                               minibatcher_fn(model.training_outputs))):

            train_err += train_fn(input_scaler_fn(input_minibatch), output_minibatch)
            model.metadata['num_minibatches'] += 1

            if (model.metadata['num_minibatches'] + 1) % model.validation_interval == 0:
                # compute validation
                validation_err = None
                if model.use_validation_routine is not None:
                    validation_err = model.use_validation_routine(model)
                elif model.validation_set is not None:
                    validation_err = 0
                    for input_minibatch, output_minibatch in itertools.izip(
                            *map(minibatcher_fn, model.validation_set)):
                        validation_err += validation_fn(input_scaler_fn(input_minibatch),
                                                        output_minibatch)

                train_err /= model.validation_interval

                # model snapshot
                elapsed_time = time.time() - start_time
                model.save_snapshot(train_err=train_err,
                                    validation_err=validation_err,
                                    elapsed_time=elapsed_time)

                # Then we print the results for this epoch:
                report = 'Training round {:.1f} secs; '.format(elapsed_time)
                report += 'training {:.6f};'.format(train_err)
                if validation_err is not None:
                    report += ' validation {:.6f}'.format(validation_err)
                if counting_epochs:
                    report += ' epochs {} / {}'.format(epoch, max_num_epochs)
                print(report)
                start_time = time.time()
                train_err = 0

            # stop when training criterion is reached
            if (max_num_minibatches is not None and
                    max_num_minibatches <= model.metadata['num_minibatches']):
                break

        # update num_epochs
        if counting_epochs:
            model.metadata['num_epochs'] += 1
