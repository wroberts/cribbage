#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
expt.py
(c) Will Roberts  12 January, 2017

Training an AI to play cribbage.
'''

from __future__ import absolute_import, print_function
import functools
import itertools
import random
from cribbage.dqlearning import DQLearner
from cribbage.game import compare_players
from cribbage.netbuilder import ModelStore, Model, build
from cribbage.neural import discard_state_repr, record_both_player_states, record_player1_states
from cribbage.player import CribbagePlayer
from cribbage.randomplayer import RandomCribbagePlayer
from cribbage.utils import doubled, numpy_memoize, random_skip
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
    for (state, _action, _reward, state2) in random_discard_sars_gen(random_seed):
        yield state
        if state2 is not None:
            yield state2

# ------------------------------------------------------------
#  Autoencode discard() states

def build_dautoenc():
    '''Construct a single-layer discard() state autoencoder.'''
    # models will be stored in the models/ directory
    #store = ModelStore('models')
    # create and configure a new model
    dautoenc = Model('models', 'dautoenc')
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
    dautoenc.max_num_minibatches(65000)
    dautoenc.validation_interval = 250 # about five minutes on samarkand
    # build the model
    build(dautoenc)

# ------------------------------------------------------------
# Two-layer discard() autoencoder

def build_dautoenc2():
    '''Construct a two-layer discard() state autoencoder.'''
    # create and configure a new model
    dautoenc2 = Model('models', 'dautoenc2')
    # network architecture
    dautoenc2.input(295)
    dautoenc2.hidden(150, 'rectify', dropout=0.2) # Dense
    dautoenc2.hidden(150, 'rectify', dropout=0.2) # Dense
    dautoenc2.output(295, 'rectify') # Dense
    dautoenc2.objective('squared_error')
    dautoenc2.update('adadelta')
    # initialise weights on first layer
    dautoenc = Model('models', 'dautoenc').load_snapshot(10000)
    dautoenc2.set_weights('hidden1', dautoenc.get_weights('hidden1'))
    # build a validation set with fixed random state
    val_set = list(itertools.islice(doubled(random_skip(random_discard_state_gen(42))), 500))
    dautoenc2.validation(val_set)
    # training stream with non-fixed random state
    stream = doubled(random_skip(random_discard_state_gen()))
    dautoenc2.training(stream)
    # configure training loop
    dautoenc2.minibatch_size(500)
    dautoenc2.max_num_minibatches(30000)
    dautoenc2.validation_interval = 250 # about five minutes on samarkand
    # build the model
    build(dautoenc2)


# ------------------------------------------------------------
#  Q-learning on discard()

def plot_training(model_name='dqlearner_a2'):
    '''Wrap code to plot the training and validation error of a given model.'''
    import matplotlib.pyplot as plt
    model = Model('models', model_name)

    plt.clf()
    data = [[ss['num_minibatches'], ss['train_err'], ss['validation_err']] for ss in
            model.metadata['snapshots']]
    data = np.array(data).T
    fig, ax1 = plt.subplots()
    ax1.plot(data[0], data[1], label='Training Error', color='C0')
    ax2 = plt.twinx()
    ax2.plot(data[0], data[2], label='Validation Error', color='C1')
    ax1.set_xlabel('Number of minibatches')
    ax1.set_ylabel('Mean squared training error per minibatch')
    ax1.tick_params('y', colors='C0')
    ax2.set_ylabel('Validation error per minibatch')
    ax2.tick_params('y', colors='C1')
    #fig.legend()
    fig.tight_layout()
    fig.show()

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
    if len(states_matrix) == 0:
        return np.array([], dtype=int)
    output = qlearner_model.compute(states_matrix)
    # only consider those actions which are possible in the given hands
    masked_output = np.ma.masked_array(output, mask=~states_matrix[:,1:53].astype(bool))
    return masked_output.argmax(axis=1)

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
        super(QLearningPlayer, self).__init__()
        self.discard_model = discard_model
        self.play_card_model = play_card_model
        self.epsilon = epsilon

    def discard(self,
                is_dealer,
                hand,
                player_score,
                opponent_score):
        '''Discard two cards from dealt hand into crib.'''
        if self.discard_model is not None and random.random() > self.epsilon:
            hand = hand[:]
            hand2 = hand[:]
            # choose the first card to discard
            state = discard_state_repr(is_dealer,
                                       hand,
                                       None,
                                       player_score,
                                       opponent_score)
            discard_value_1 = get_best_actions(self.discard_model, state[None, :])[0]
            discard_idx_1 = hand.index(discard_value_1)
            # remove the first discard from the hand and re-encode
            del hand[discard_idx_1]
            state = discard_state_repr(is_dealer,
                                       hand,
                                       discard_value_1,
                                       player_score,
                                       opponent_score)
            discard_value_2 = get_best_actions(self.discard_model, state[None, :])[0]
            discard_idx_2 = hand2.index(discard_value_2)
            return [discard_idx_1, discard_idx_2]
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
        '''Select a single card from the hand to play during cribbage play.'''
        if self.play_card_model is not None and random.random() > self.epsilon:
            # TODO: integrate self.play_card_model
            pass
        return random.choice(legal_moves)

def dqlearner_vs_random(qlearner_model, _dummy_model):
    '''
    Plays a set of games between the Q-Learner player and a
    RandomPlayer, returns the fraction that the Q-Learner player wins.

    Arguments:
    - `qlearner_model`: a Model object
    '''
    qplayer = QLearningPlayer(qlearner_model, None, epsilon=0.05)
    stats = compare_players([qplayer, RandomCribbagePlayer()], 100)
    return stats[0] / 100.

@numpy_memoize(ModelStore('models', ensure_exists=True).join('discard_scaling.npz'))
def get_discard_scaling():
    '''
    Estimates the mean and standard deviation of input vectors to the
    discard() neural network.
    '''
    inputs = np.array(list(itertools.islice(random_discard_state_gen(), 100000)))
    return inputs.mean(axis=0), inputs.std(axis=0)

def make_discard_input_scaler():
    '''
    Return a function which can scale an input vector (or array of
    input vectors) to a normal distribution, given the population mean
    and standard deviation.
    '''
    mean, std = get_discard_scaling()
    def discard_input_scaler(inputs):
        '''Zero-centre and normalise a matrix of inputs.'''
        return (inputs - mean) / std
    return discard_input_scaler

# Q-learning model for discard()
def make_dqlearner(store, name):
    '''
    Builds a Q-learning model to learn how to discard.

    Arguments:
    - `store`: the ModelStore to store the Model in
    - `name`: the name of the Q-learning model to create
    '''
    model = Model(store, name)
    model.input(347)
    model.hidden(150, 'rectify') # Dense
    model.hidden(150, 'rectify') # Dense
    model.output(52, 'linear') # Dense: top two activations indicate cards to play
    model.objective('squared_error')
    model.update('rmsprop')
    model.update_args({'learning_rate': 0.002})
    # normalise inputs to network
    model.input_scaler(make_discard_input_scaler())
    # initialise weights from dautoenc2
    #dautoenc2 = Model(store, 'dautoenc2').load_snapshot(20000)
    #model.set_weights('hidden1', dautoenc2.get_weights('hidden1'))
    #model.set_weights('hidden2', dautoenc2.get_weights('hidden2'))
    # validation will be performed by playing cribbage against a random
    # player
    model.minibatch_size(32)
    model.validation_interval = 6000
    return model

def record_player1_discard_sars_gen(model, epsilon):
    '''
    Returns an infinite generator of (s,a,r,s2) discard tuples by
    playing the given model with the given epsilon value against a
    random player.

    Arguments:
    - `model`:
    - `epsilon`:
    '''
    while True:
        discard_states, _ = record_player1_states(
            QLearningPlayer(model, None, epsilon=epsilon),
            RandomCribbagePlayer())
        for state in discard_states:
            yield state

# ------------------------------------------------------------
#  Double Q-Learning

# build the two q-learning networks
dqlearner_a = make_dqlearner('models', 'dqlearner_a9')
dqlearner_a.validation_routine(functools.partial(dqlearner_vs_random, dqlearner_a))
dqlearner_b = make_dqlearner('models', 'dqlearner_b9')
dqlearner_b.validation_routine(functools.partial(dqlearner_vs_random, dqlearner_a))

learner = DQLearner(dqlearner_a, dqlearner_b,
                    random_discard_sars_gen,
                    record_player1_discard_sars_gen)
# initialise replay memory with 50,000 (s,a,r,s) tuples from random play
learner.replay_memory_init_size(50000)
# 50k: 252M
# 100k: 360M
# 150k: 353M
# 200k: 414M
# 500k: 750M
# truncate replay memory at 500K (replay memory was 1M states in Mnih)
learner.replay_memory_max_size(500000)
# e-greedy with epsilon annealed linearly from 1.0 to 0.1 over first
# 1,000,000 minibatches, and 0.1 thereafter
learner.epsilon_fn(lambda n: max(1. + (0.1 - 1.) * n / 1000000., 0.1))
# on every training loop, sample 5K (s,a,r,s) discard states and store
# in the replay memory
learner.samples_per_loop(5000)
# make the training set 312 random minibatches (sampling with
# replacement) of 32 s,a,r,s tuples (this is roughly in line with
# Mnih's "Qhat estimator updated every 10,000 updates")
learner.minibatch_size(32)
learner.minibatches_per_loop(312)
learner.choose_action_fn(get_best_actions)
learner.train()

# In [8]: cProfile.run('loop(replay_memory, dqlearner_a, dqlearner_b)', sort='time')
#          822065 function calls in 2.806 seconds
#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#       315    0.921    0.003    1.523    0.005 function_module.py:754(__call__)
#      2193    0.553    0.000    0.553    0.000 {numpy.core.multiarray.dot}
#      2505    0.187    0.000    0.593    0.000 round.py:144(play)
#       315    0.152    0.000    0.152    0.000 expt.py:258(discard_input_scaler)
#     29986    0.100    0.000    0.100    0.000 {numpy.core.multiarray.zeros}
#      2506    0.079    0.000    0.347    0.000 round.py:68(deal)
#      2506    0.076    0.000    0.089    0.000 random.py:277(shuffle)
#      9981    0.073    0.000    0.164    0.000 neural.py:111(play_state_repr)
#      2827    0.060    0.000    0.060    0.000 {numpy.core.multiarray.array}
#      9981    0.036    0.000    0.293    0.000 neural.py:268(play_card)
#     24974    0.035    0.000    0.035    0.000 neural.py:41(encode_categories)
#     22469    0.031    0.000    0.039    0.000 random.py:273(choice)
#     64234    0.031    0.000    0.031    0.000 neural.py:28(one_hot)
