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
from cribbage.utils import doubled
import numpy as np

def random_discard_sars_gen(random_seed=None):
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

def random_discard_state_gen(random_seed=None):
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
dautoenc.update_args({}) # 'learning_rate': 1.0, 'rho': 0.95, 'epsilon': 1e-6
# build a validation set with fixed random state
val_set = list(itertools.islice(doubled(random_skip(random_discard_state_gen(42))), 500))
dautoenc.validation(val_set)
# training stream with non-fixed random state
stream = doubled(random_skip(random_discard_state_gen()))
dautoenc.training(stream)
# configure training loop
dautoenc.minibatch_size(500)
dautoenc.num_minibatches(10000)
dautoenc.validation_interval = 250 # about five minutes on samarkand
# build the model
build(dautoenc)

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
dautoenc = Model(store, 'dautoenc').load_snapshot(10000)
dautoenc2.set_weights('hidden1', dautoenc.get_weights('hidden1'))
# build a validation set with fixed random state
val_set = list(itertools.islice(doubled(random_skip(random_discard_state_gen(42))), 500))
dautoenc2.validation(val_set)
# training stream with non-fixed random state
stream = doubled(random_skip(random_discard_state_gen()))
dautoenc2.training(stream)
# configure training loop
dautoenc2.minibatch_size(500)
dautoenc2.num_minibatches(10000)
dautoenc2.validation_interval = 250 # about five minutes on samarkand
# build the model
build(dautoenc2)

import matplotlib.pyplot as plt
model = dautoenc2

a = [[ss['num_minibatches'], ss['train_err'], ss['validation_err']] for ss in
     model.metadata['snapshots']]
a = np.array(a)
plt.plot(a.T[0], a.T[1], label='Training Error')
plt.plot(a.T[0], a.T[2], label='Validation Error')
plt.xlabel('Number of minibatches')
plt.ylabel('Mean squared error per minibatch')
plt.legend()
plt.show()

def compare_dqlearner_to_random_player(qlearner_model):
    '''
    Plays a set of games between the Q-Learner player and a
    RandomPlayer, returns the fraction that the Q-Learner player wins.

    Arguments:
    - `qlearner_model`: a Model object
    '''
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
    dautoenc2 = Model(store, 'dautoenc2').load_snapshot(12000)
    model.set_weights('hidden1', dautoenc2.get_weights('hidden1'))
    model.set_weights('hidden2', dautoenc2.get_weights('hidden2'))
    # validation will be performed by playing cribbage against a random
    # player
    model.validation_routine(compare_dqlearner_to_random_player)
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
    action_values = qlearner_model.compute(states_matrix)
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
