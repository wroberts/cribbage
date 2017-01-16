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
dautoenc.num_minibatches(65000)
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
dautoenc2.num_minibatches(30000)
dautoenc2.validation_interval = 250 # about five minutes on samarkand
# build the model
build(dautoenc2)

import matplotlib.pyplot as plt
model = Model(store, 'dautoenc2')

a = [[ss['num_minibatches'], ss['train_err'], ss['validation_err']] for ss in
     model.metadata['snapshots']]
a = np.array(a)
plt.plot(a.T[0], a.T[1], label='Training Error')
plt.plot(a.T[0], a.T[2], label='Validation Error')
plt.xlabel('Number of minibatches')
plt.ylabel('Mean squared error per minibatch')
plt.legend()
plt.show()

class QLearningPlayer(CribbagePlayer):
    '''A CribbagePlayer that plays using a Q-learned model.'''

    def __init__(self, discard_model, play_card_model, epsilon):
        '''
        Constructor.

        Arguments:
        - `discard_model`:
        - `play_card_model`:
        - `epsilon`:
        '''
        self.discard_model = discard_model
        self.play_card_model = play_card_model
        self.epsilon = epsilon

    def discard(self,
                is_dealer,
                hand,
                player_score,
                opponent_score):
        if self.discard_model is not None:
            # TODO
            pass
        return random.sample(range(6), 2)

    def play_card(self,
                  is_dealer,
                  hand,
                  played_cards,
                  is_go,
                  linear_play,
                  player_score,
                  opponent_score,
                  legal_moves):
        if self.play_card_model is not None:
            # TODO
            pass
        return random.choice(legal_moves)

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
    dautoenc2 = Model(store, 'dautoenc2').load_snapshot(20000)
    model.set_weights('hidden1', dautoenc2.get_weights('hidden1'))
    model.set_weights('hidden2', dautoenc2.get_weights('hidden2'))
    # validation will be performed by playing cribbage against a random
    # player
    model.validation_routine(compare_dqlearner_to_random_player)
    model.minibatch_size(32)
    model.num_epochs(1)
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
    output = qlearner_model.compute(states_matrix)
    # only consider those actions which are possible in the given hands
    masked_output = np.ma.masked_array(output, mask=~states_matrix[:,1:53].astype(bool))
    return masked_output.argmax(axis=1)

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
    return qlearner_model.compute(states_matrix)[np.arange(len(actions_vector)),
                                                 actions_vector]

# initialise replay memory with 50,000 (s,a,r,s) tuples from random play
replay_memory = []
replay_memory.extend(itertools.islice(random_discard_sars_gen(), 50000))
# 50k: 252M
# 100k: 360M
# 150k: 353M
# 200k: 414M
# 500k: 750M
# build the two q-learning networks
dqlearner_a = make_dqlearner(store, 'dqlearner_a')
dqlearner_b = make_dqlearner(store, 'dqlearner_b')
# randomly select which q-learning network will be updated, and which
# will estimate action values
if random.random() < 0.5:
    dqlearner_update = dqlearner_a
    dqlearner_scorer = dqlearner_b
else:
    dqlearner_update = dqlearner_b
    dqlearner_scorer = dqlearner_a
# play games against the random player and record (s,a,r,s) discard
# states until 10,000 discard states have been generated
#
# e-greedy with epsilon annealed linearly from 1.0 to 0.1 over first
# 1,000,000 "frames", and 0.1 thereafter
record_player1_states(QLearningPlayer(dqlearner_update), RandomCribbagePlayer())
# add these to the replay memory
pass # TODO
# truncate replay memory if needed (replay memory was 1,000,000 states in Mnih)
if len(replay_memory) > 500000:
    replay_memory = replay_memory[-500000:]
# make the training set 312 random minibatches (sampling with
# replacement) of 32 s,a,r,s tuples (this is roughly in line with
# Mnih's "Qhat estimator updated every 10,000 updates")
selected_idxs = np.random.randint(0, len(replay_memory), size=312*32)
selected_sars = [replay_memory[idx] for idx in selected_idxs]
pre_states = np.array([s for s,a,r,s2 in selected_sars])
actions = np.array([a for s,a,r,s2 in selected_sars]).argmax(axis=1)
rewards = np.array([r for s,a,r,s2 in selected_sars])
# handle cases where post_state is None: keep track of indices into
# our matrices (e.g., pre_states, actions) where the post_state is not
# None
nonnull_post_state_idxs = np.array([i for i,(s,a,r,s2) in enumerate(selected_sars) if s2 is not None])
post_states = np.array([s2 for s,a,r,s2 in selected_sars if s2 is not None])
# the online q-learner is used to figure out what the optimal future
# actions will be
best_actions = get_best_actions(dqlearner_update, post_states)
# and the other q-learner is used to score the values of these future actions
value_estimates = get_scores(dqlearner_scorer, post_states, best_actions)
# we compute the action-value matrix for the pre-states from
# dqlearner_update
previous_values = dqlearner_update.compute(pre_states)
# the updated values for training is identical to previous_values,
# except for at the locations of the selected actions, which are set
# to the new estimates of those action values.  the SGD of the neural
# network will take care of nudging the weights towards these new
# values (i.e., we do not use the "alpha" from van Hasselt's poster.
gamma = 0.99
updated_values = np.array(previous_values)
updated_values[np.arange(len(actions)), actions] = rewards
# in cases where the post_state is None, the value_estimate for that
# post_state is defined to be 0
updated_values[nonnull_post_state_idxs, actions[nonnull_post_state_idxs]] += gamma * value_estimates
# train updated values for one epoch
dqlearner_update.training((pre_states, updated_values))
