#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
expt.py
(c) Will Roberts  12 January, 2017

Training an AI to play cribbage.
'''

from __future__ import absolute_import, print_function
import itertools
import random
from cribbage.game import Game
from cribbage.netbuilder import ModelStore, Model, build
from cribbage.neural import record_both_player_states, record_player1_states
from cribbage.randomplayer import RandomCribbagePlayer
from cribbage.simpleplayer import SimpleCribbagePlayer
from cribbage.utils import doubler, mkdir_p, open_atomic
import lasagne
import numpy as np
import theano
import theano.tensor as T

def random_discard_sars_gen(random_seed = None):
    '''
    Infinite generator over discard (state, action, reward,
    next_state) tuples, using a random player.  Produces about 2700
    states per second on samarkand.

    Arguments:
    - `random_seed`:
    '''
    random.seed(random_seed)
    player = RandomCribbagePlayer()
    while True:
        discard_states1, _pcs1, discard_states2, _pcs2 = record_both_player_states(player, player)
        for state in discard_states1:
            yield state
        for state in discard_states2:
            yield state

def random_discard_state_gen(random_seed = None):
    '''
    Infinite generator over discard states, using a random player.
    Produces about 2700 states per second on samarkand.

    Arguments:
    - `random_seed`:
    '''
    for (state, _a, _r, _s) in random_discard_sars_gen(random_seed):
        yield state

def random_skip(seq, p=0.2):
    '''
    Generator yields items from a sequence, randomly skipping some of them.

    Arguments:
    - `seq`:
    - `p`: the probability of emitting an item
    '''
    for item in seq:
        if random.random() < p:
            yield item

# http://stackoverflow.com/a/8991553/1062499
def grouper(n, iterable):
    it = iter(iterable)
    while True:
       chunk = tuple(itertools.islice(it, n))
       if not chunk:
           return
       yield chunk

def minibatcher(n, iterable):
    for item in grouper(n, iterable):
        yield np.array(item)

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
dautoenc.update_params({}) # 'learning_rate': 1.0, 'rho': 0.95, 'epsilon': 1e-6
# build a validation set with fixed random state
val_set = list(itertools.islice(doubler(random_skip(random_discard_state_gen(42))), 500))
dautoenc.validation(val_set)
# training stream with non-fixed random state
stream = doubler(random_skip(random_discard_state_gen()))
dautoenc.training(stream)
# configure training loop
dautoenc.minibatch_size(500)
dautoenc.num_minibatches(10000)
# build the model
build(dautoenc)

# implementation of build

data = T.matrix('data')
predictions = lasagne.layers.get_output(dautoenc.network, data)
targets = T.matrix('targets')
loss = lasagne.objectives.squared_error(predictions, targets)
loss = lasagne.objectives.aggregate(loss, mode='mean')
validation_predictions = lasagne.layers.get_output(dautoenc.network, data, deterministic=True)
validation_loss = lasagne.objectives.squared_error(validation_predictions, targets)
validation_loss = lasagne.objectives.aggregate(validation_loss, mode='mean')

if dautoenc.minibatch_size_value is not None:
    minibatcher_fn = lambda xs: minibatcher(dautoenc.minibatch_size_value, xs)
else:
    minibatcher_fn = lambda xs: xs
training_inputs = minibatcher_fn(dautoenc.training_inputs)
training_outputs = minibatcher_fn(dautoenc.training_outputs)

params = lasagne.layers.get_all_params(dautoenc.network, trainable=True)
update_fn = UPDATE_NAMES[dautoenc.update_name]
updates = update_fn(loss, params, **dautoenc.update_params_value)

train_fn = theano.function([data, targets], loss, updates=updates)

validation_fn = theano.function([data, targets], validation_loss)

import time
start_time = time.time()

train_err = 0
train_minibatches = 0
for num_minibatches, (input_minibatch, output_minibatch) in enumerate(
        itertools.izip(training_inputs, training_outputs)):

    train_err += train_fn(input_minibatch, output_minibatch)
    train_minibatches += 1

    dautoenc.validation_interval = 1
    if (num_minibatches + 1) % dautoenc.validation_interval == 0:
        # compute validation
        validation_err = 0
        for input_minibatch, output_minibatch in itertools.izip(*map(minibatcher_fn, dautoenc.validation_set)):
            validation_err += validation_fn(input_minibatch, output_minibatch)

        # Then we print the results for this epoch:
        print('Last training round took {:.3f}s'.format(time.time() - start_time))
        print('  training loss:\t\t{:.6f}'.format(train_err / dautoenc.validation_interval))
        print('  validation loss:\t\t{:.6f}'.format(validation_err))
        start_time = time.time()
        train_err = 0

# ------------------------------------------------------------
# Notes

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
val_set = list(itertools.islice(doubler(random_skip(random_discard_state_gen(42)), 500)))
dautoenc2.validation(val_set)
# training stream with non-fixed random state
stream = doubler(random_skip(random_discard_state_gen()))
dautoenc2.training(stream)
# configure training loop
dautoenc2.minibatch_size(500)
dautoenc2.num_minibatches(10000)
# build the model
build(dautoenc2)

def compare_dqlearner_to_random_player(qlearner_model):
    pass # TODO

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
