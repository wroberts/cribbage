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
from cribbage.netbuilder import ModelStore, Model
from cribbage.utils import doubler, mkdir_p, open_atomic
import lasagne
import numpy as np
import theano
import theano.tensor as T

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
