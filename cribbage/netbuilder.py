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
import os
import random
from cribbage.utils import doubler, mkdir_p, open_atomic
import lasagne
import numpy as np
import theano
import theano.tensor as T

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
    'categorical_crossentropy': lasagne.objectives.categorical_crossentropy,
    'squared_error': lasagne.objectives.squared_error,
    }

class Model(object):
    '''An object wrapping a Lasagne feedforward neural network.'''

    MAX_VALIDATION_SET_SIZE = 10000

    def __init__(self, store, model_name):
        '''
        Constructor

        Arguments:
        - `store`:
        - `model_name`:
        '''
        self.store = store
        self.model_name = model_name
        self.objective_name = 'squared_error'
        self.update_name = 'adadelta'
        # validation is computed after this many minibatches have been
        # trained
        self.validation_interval = 50
        # this is a tuple of (np.array, np.array) representing inputs
        # and outputs; if it is not None, it is used between
        # minibatches to compute validation
        self.validation_set = None
        # if not None, this function is passed this object to compute
        # a validation statistic
        self.use_validation_fn = None
        # these are iterables of np.array values representing inputs
        # and outputs; if not None, they are used for training
        self.training_inputs = None
        self.training_outputs = None
        # minibatch size; if this is not None, training (input,
        # output) pairs are grouped into blocks of this size during
        # training
        self.use_minibatch_size = None
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
        # if the architecture is loaded from metadata, the input(),
        # hidden() and output() methods are used to check; this
        # variable stores which layer we're currently checking
        self.arch_check_stage = 0
        # this flag indicates if the network architecture has been
        # fully specified (for new Models) or has been fully checked
        # (for Models loaded from disk)
        self.arch_desc_complete = False
        # this variable holds the actual neural network
        self._network = None
        # load metadata if possible
        self.ensure_exists()
        try:
            self.load_metadata()
            if 'architecture' in self.metadata:
                self.arch_frozen = True
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

    def save_metadata(self):
        '''Saves metadata for this Model to disk.'''
        with open_atomic(self.metadata_filename, 'wb') as output_file:
            output_file.write(json.dumps(self.metadata, indent=4))

    @property
    def best_validation_error(self):
        pass # TODO

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

    def _build_network(self):
        '''
        Builds the Lasagne network for this Model from the description in
        metadata['architecture'].
        '''
        assert 'architecture' in self.metadata
        assert len(self.metadata['architecture']) > 1
        assert self.metadata['architecture'][0]['layer'] == 'input'
        assert self.metadata['architecture'][-1]['layer'] == 'output'
        if self._network is None:
            num_hidden_layers = 0
            for layer in self.metadata['architecture']:
                if layer['layer'] == 'input':
                    assert self._network is None
                    self._network = lasagne.layers.InputLayer(shape=(None, layer['size']),
                                                              name='input')
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
    def network(self):
        '''Returns this Model's neural network object.'''
        if not self._network:
            self._build_network()
        return self._network

    def set_weights(self, param_name, param_set):
        pass # TODO

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
            inputs = np.array([i for (i,o) in validation_set])
            outputs = np.array([o for (i,o) in validation_set])
            self.validation_set = (inputs, outputs)

    def validation_fn(self, validation_fn):
        '''
        Sets the validation function used for validation during training.
        The argument should be a function which, when passed this
        object, returns a validation statistic.

        Arguments:
        - `validation_fn`:
        '''
        self.use_validation_fn = validation_fn

    def training(self, training_set):
        '''
        Sets the training set used for training.

        Arguments:
        - `training_set`:
        '''
        # training set might be a tuple: (inputs, outputs)
        if isinstance(training_set, tuple) and len(training_set) == 2:
            self.training_inputs, self.training_outputs = training_set
        else:
            # otherwise, it might be an iterable
            #
            # the iterable is taken to consist of tuples of (input,
            # output) pairs
            inputs, outputs = itertools.tee(training_set)
            self.training_inputs = (i for (i,o) in inputs)
            self.training_outputs = (o for (i,o) in outputs)

    def minibatch_size(self, minibatch_size):
        '''
        Sets the size of minibatches to use for training.

        Arguments:
        - `minibatch_size`:
        '''
        self.use_minibatch_size = minibatch_size

    def num_minibatches(self, num_minibatches):
        '''Configures how long training should run for.  Useful if the training set is infinitely long.

        Arguments:
        - `num_minibatches`:
        '''
        self.use_num_minibatches = num_minibatches

    def num_epochs(self, num_epochs):
        '''Configures how long training should run for, if the training set is of finite length.

        Arguments:
        - `num_epochs`:
        '''
        self.use_num_epochs = num_epochs


dautoenc.network
data = T.matrix('data')
predictions = lasagne.layers.get_output(dautoenc.network)
targets = T.matrix('targets')
loss = lasagne.objectives.squared_error(predictions, targets)
loss = lasagne.objectives.aggregate(loss, mode='mean')
validation_predictions = lasagne.layers.get_output(dautoenc.network, data, deterministic=True)
validation_loss = lasagne.objectives.squared_error(validation_predictions, targets)
validation_loss = lasagne.objectives.aggregate(validation_loss, mode='mean')

# models will be stored in the models/ directory
store = ModelStore('models')

# create and configure a new model
dautoenc = Model(store, 'dautoenc')
# network architecture
dautoenc.input(295)
dautoenc.hidden(150, 'rectify', dropout=0.2) # Dense
dautoenc.output(295, 'rectify') # Dense
dautoenc.objective('squared_error')
dautoenc.update('adadelta')
# build a validation set with fixed random state
val_set = list(itertools.islice(doubler(random_discard_state_gen(42)), 500))
dautoenc.validation(val_set)
# training stream with non-fixed random state
stream = doubler(random_discard_state_gen())
dautoenc.training(stream)
# configure training loop
dautoenc.minibatch_size(500)
dautoenc.num_minibatches(10000)
# build the model
build(dautoenc)

# create and configure a new model
dautoenc2 = Model(store, 'dautoenc2')
# network architecture
dautoenc2.input(295)
dautoenc2.hidden(150, 'rectify', dropout=0.2) # Dense
dautoenc2.hidden(150, 'rectify', dropout=0.2) # Dense
dautoenc2.output(295, 'rectify') # Dense
dautoenc2.objective('squared_error')
dautoenc2.update('adadelta')
# initialise weights on first layer
dautoenc = Model(store, 'dautoenc').best_validation_error
dautoenc2.set_weights('hidden1', dautoenc.get_weights('hidden1'))
# build a validation set with fixed random state
val_set = list(itertools.islice(doubler(random_discard_state_gen(42)), 500))
dautoenc2.validation(val_set)
# training stream with non-fixed random state
stream = doubler(random_discard_state_gen())
dautoenc2.training(stream)
# configure training loop
dautoenc2.minibatch_size(500)
dautoenc2.num_minibatches(10000)
# build the model
build(dautoenc2)

# Q-learning model for discard()
def make_dqlearner(store, name):
    model = Model(store, name)
    model.input(295)
    model.hidden(150, 'rectify', dropout=0.2) # Dense
    model.hidden(150, 'rectify', dropout=0.2) # Dense
    model.output(52, 'rectify') # Dense: top two activations indicate cards to play
    model.objective('squared_error')
    model.update('adadelta')
    # initialise weights from dautoenc2
    dautoenc2 = Model(store, 'dautoenc2').best_validation_error
    model.set_weights('hidden1', dautoenc2.get_weights('hidden1'))
    model.set_weights('hidden2', dautoenc2.get_weights('hidden2'))
    # validation will be performed by playing cribbage against a random
    # player
    model.validation_fn(compare_dqlearner_to_random_player)
    model.num_epochs(5)
    return model

def get_best_actions(qlearner_model, states_matrix):
    '''
    Given a Model with a Q-learning neural network in it, and a matrix
    of N states, returns a vector of length N containing the argmax of
    the network's outputs for each state (the action the network would
    choose in that state).

    Arguments:
    - `qlearner_model`:
    - `states_matrix`:
    '''
    return None # TODO

def get_scores(qlearner_model, states_matrix, actions_vector):
    '''
    Given a Model with a Q-learning neural network in it, and a matrix
    of N states, and a vector of N integer values (each of which can
    be one-hot encoded to action vectors), returns a vector of length
    N giving the network's valuation of those state-action pairs.

    Arguments:
    - `qlearner_model`:
    - `states_matrix`:
    - `actions_vector`:
    '''
    return None # TODO

dqlearner_a = make_dqlearner(store, 'dqlearner_a')
dqlearner_b = make_dqlearner(store, 'dqlearner_b')
# training will be done with online policy updating
# randomly select action-chooser network and action-value-estimator network
if random.random() < 0.5:
    dqlearner_update = dqlearner_a
    dqlearner_scorer = dqlearner_b
else:
    dqlearner_update = dqlearner_b
    dqlearner_scorer = dqlearner_a
# play a bunch of games against random player and record (s,a,r,s) discard states
record_player1_states(QLearningPlayer(dqlearner_update), RandomCribbagePlayer())
# randomly select a bunch of these states
selected_sars = BLAH
pre_states, actions, rewards, post_states = selected_sars
# calculate values of (s,a) using the action-value-estimator
best_actions = get_best_actions(dqlearner_update, post_states)
value_estimates = get_scores(dqlearner_scorer, post_states, best_actions)
previous_values = get_scores(dqlearner_update, pre_states, actions)
# update those values
updated_values = previous_values + alpha * (rewards + gamma * value_estimates - previous_values)
# train updated values
dqlearner_update.training((pre_states, updated_values))
