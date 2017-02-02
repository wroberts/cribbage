#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
dqlearning.py
(c) Will Roberts   2 February, 2017

Double Q-Learning API

We assume that Q-learning is accomplished using a model that takes
one or more state vectors and produces one or more Q-value vectors,
each of length M, which record the model's estimate of the value of
each of the M actions that are possible in the given task.

The double Q-learner works using samples stored in a replay memory.
These samples are tuples, called (s,a,r,s) tuples.  These store four
items: an initial state (s), an action (a), a reward (r), and a final
state (s2).  States (s and s2) are taken to be vectors of constant
length.  Actions are integer values between 0 and M-1 inclusive.
Rewards are floating point values, and care should be taken to ensure
that these fall in the range between -1.0 and 1.0.

If the environment is episodic, certain actions may result in the end
of an episode.  In these cases, the final state (s2) vector should be
set to the value None.  Such None states are assumed to have a
constant Q-value of 0.
'''

from __future__ import absolute_import
import itertools
import random
import numpy as np
from cribbage.netbuilder import build

def default_choose_action_func(model, states):
    '''
    Default adaptor method for greedily choosing an action from a
    given state.

    Arguments:
    - `model`: a Q-learning model
    - `states`: a matrix containing zero or more state vectors
    '''
    if len(states) == 0:
        return np.array([])
    q_values = model.compute(states)
    return q_values.argmax(axis=1)

def default_q_value_func(model, states, actions=None):
    '''
    Default adaptor method for computing the Q-values for a given set
    of states.

    Given a Model with a Q-learning neural network in it, and a matrix
    of N states, returns a matrix of length N giving the network's
    valuation of all the actions in each of those states.  Optionally,
    a vector of N integer values can be given (each of which
    represents an action taken in each state), in which case this
    function returns a vector of length N, giving the network's
    valuation of those state-action pairs.

    Arguments:
    - `model`: a Q-learning model
    - `states`: a matrix containing zero or more state vectors
    - `actions`: optionally, a vector containing the actions taken in
      each state of `states`
    '''
    if len(states) == 0:
        return np.array([])
    q_values = model.compute(states)
    if actions is not None:
        q_values = q_values[np.arange(len(actions)), actions]
    return q_values

class DQLearner(object):
    '''A class implementing double Q-learning.'''

    def __init__(self, model_a, model_b, init_sars_fn, sample_sars_fn):
        '''
        Constructor.

        Arguments:
        - `model_a`:
        - `model_b`:
        - `init_sars_fn`:
        - `sample_sars_fn`:
        '''
        self.model_a = model_a
        self.model_b = model_b
        self.rmem_init_size = 50000
        self.rmem_max_size = 500000
        self.nsamples_per_loop = 5000
        self.nminibatch_size = 32
        self.nminibatches_per_loop = 100
        self.gamma_value = 0.99
        self.init_sars_func = init_sars_fn
        self.init_sars_epsilon_value = 0.1
        self.sample_sars_func = sample_sars_fn
        self.epsilon_func = lambda n: 0.1
        self.choose_action_func = default_choose_action_func
        self.q_value_func = default_q_value_func

    def replay_memory_init_size(self, num):
        '''
        Sets the number of (s,a,r,s) tuples that the replay memory should
        start with, before training begins.  Defaults to 50,000.

        Arguments:
        - `num`:
        '''
        self.rmem_init_size = num

    def replay_memory_max_size(self, num):
        '''
        Sets the maximum number of (s,a,r,s) tuples that the replay memory
        should contain.  Defaults to 500,000.

        Arguments:
        - `num`:
        '''
        self.rmem_max_size = num

    def samples_per_loop(self, num):
        '''
        Sets the number of (s,a,r,s) tuples that should be sampled during
        each loop of the train() function.  Defaults to 5,000.

        Arguments:
        - `num`:
        '''
        self.nsamples_per_loop = num

    def minibatch_size(self, num):
        '''
        Sets the minibatch size.  Defaults to 32.

        Arguments:
        - `num`:
        '''
        self.nminibatch_size = num

    def minibatches_per_loop(self, num):
        '''
        Sets how many minibatches will be sampled from the replay_memory
        and trained on the model during each loop of the train()
        function.  Defaults to 100.

        Arguments:
        - `num`:
        '''
        self.nminibatches_per_loop = num

    def gamma(self, value):
        '''
        Sets the value of gamma used in the Q-learning update.  Defaults
        to 0.99.

        Arguments:
        - `value`:
        '''
        self.gamma_value = value

    def init_sars_fn(self, func):
        '''
        This sets a function which is used to initialise the replay memory
        when train() is first called.  This function should take no
        arguments and return a generator over (s,a,r,s) tuples;
        typically, you will want this function to produce tuples
        according to a random policy.

        for (s,a,r,s2) in init_sars_func():
            pass

        Arguments:
        - `func`:
        '''
        self.init_sars_func = func

    def init_sars_epsilon(self, value):
        '''
        Sets the epsilon value used to produce samples for initialising
        the replay memory when train() is first called.  If a
        DQLearner object is constructed with models that have
        previously been trained, the replay memory is initialised by
        sampling from one of the models, using this epsilon value.
        Defaults to 0.1.

        Arguments:
        - `value`:
        '''
        self.init_sars_epsilon_value = value

    def sample_sars_fn(self, func):
        '''
        This sets a function which is used to add (s,a,r,s) tuples to the
        replay memory when the train() function is running.  The
        function should take a `model` and `epsilon` parameters, and
        produce a generator over (s,a,r,s) tuples.

        for (s,a,r,s2) in sample_sars_func(model, epsilon):
            pass

        Arguments:
        - `func`:
        '''
        self.sample_sars_func = func

    def epsilon_fn(self, func):
        '''
        This sets a function which updates the epsilon value used for
        sampling.  The function should take a single `n` parameter,
        representing the number of minibatches that the model in
        question has already been trained for:

        epsilon = epsilon_func(model.num_minibatches)

        Defaults to a constant function that always returns 0.1

        Arguments:
        - `func`:
        '''
        self.epsilon_func = func

    def choose_action_fn(self, func):
        '''
        This sets an adaptor method which greedily selects the optimal
        action for a given state.  The function should take a `model`
        parameter, and a `states` matrix, and return an `actions`
        vector containing the integer values of the optimal actions
        for each state in `states`.

        actions = choose_action_func(model, states)

        Arguments:
        - `func`:
        '''
        self.choose_action_func = func

    def q_value_fn(self, func):
        '''
        This sets an adaptor method which finds the Q-values for all
        actions in a given state.  The function should take a `model`
        parameter, and a `states` matrix, as well as an optional
        `actions` vector.  If `actions` is not given, it returns a
        matrix of Q-values, where each row corresponds to one of the
        states in `states, and is of length M, specifying the Q-value
        of taking each possible action in that states.  If `actions`
        is given (as a vector of integer action values), the function
        returns a vector, specifying the Q-value of taking the given
        action in each state.

        q_values = q_value_func(model, states, actions=None)

        Arguments:
        - `func`:
        '''
        self.q_value_func = func

    def train(self):
        '''Runs the double Q-learning algorithm to train the models.'''
        # replay memory contains (s,a,r,s) tuples
        if self.model_a.weights_loaded:
            # generate states from self.model_a
            init_gen = self.sample_sars_func(self.model_a,
                                             self.init_sars_epsilon_value)
        else:
            # intialise replay memory from random policy
            init_gen = self.init_sars_func()
        replay_memory = list(itertools.islice(init_gen, self.rmem_init_size))

        # training loop
        while True:
            # randomly select which q-learning network will be
            # updated, and which will estimate action values
            if random.random() < 0.5:
                update_model = self.model_a
                scoring_model = self.model_b
            else:
                update_model = self.model_b
                scoring_model = self.model_a
            # e-greedy policy; using epsilon_func, the epsilon value
            # can be annealed over the training regime
            epsilon = self.epsilon_func(update_model.num_minibatches)
            # sample nsamples_per_loop (s,a,r,s) tuples using an
            # e-greedy policy, and add these to the replay memory
            replay_memory.extend(
                itertools.islice(self.sample_sars_func(update_model, epsilon),
                                 self.nsamples_per_loop))
            # truncate replay memory if needed
            if len(replay_memory) > self.rmem_max_size:
                replay_memory = replay_memory[-self.rmem_max_size:]
            # make the training set self.nminibatches_per_loop random
            # minibatches (sampling with replacement) of
            # self.nminibatch_size (s,a,r,s) tuples
            selected_idxs = np.random.randint(
                0, len(replay_memory),
                size=self.nminibatches_per_loop * self.nminibatch_size)
            selected_sars = [replay_memory[idx] for idx in selected_idxs]
            pre_states = np.array([s for s,a,r,s2 in selected_sars])
            actions = np.array([a for s,a,r,s2 in selected_sars]).argmax(axis=1)  # TODO
            rewards = np.array([r for s,a,r,s2 in selected_sars])
            # handle cases where post_state is None: keep track of
            # indices into our matrices (e.g., pre_states, actions)
            # where the post_state is not None
            nonnull_post_state_idxs = np.array([i for i,(s,a,r,s2) in
                                                enumerate(selected_sars)
                                                if s2 is not None], dtype=int)
            post_states = np.array([s2 for s,a,r,s2 in selected_sars if s2 is not None])
            # the online q-learner is used to figure out what the
            # optimal future actions will be
            best_actions = self.choose_action_func(update_model, post_states)
            # and the other q-learner is used to score the values of
            # these future actions
            value_estimates = self.q_value_func(scoring_model, post_states, best_actions)
            # we compute the action-value matrix for the pre-states
            # from update_model
            previous_values = self.q_value_func(update_model, pre_states)
            # the updated values for training is identical to
            # previous_values, except for at the locations of the
            # selected actions, which are set to the new estimates of
            # those action values.  the SGD of the neural network will
            # take care of nudging the weights towards these new
            # values (i.e., we do not use the "alpha" from van
            # Hasselt's poster).
            updated_values = np.array(previous_values)
            updated_values[np.arange(len(actions)), actions] = rewards
            # in cases where the post_state is None, the
            # value_estimate for that post_state is defined to be 0
            updated_values[nonnull_post_state_idxs,
                           actions[nonnull_post_state_idxs]] += (self.gamma_value *
                                                                 value_estimates)
            # train updated values for one epoch
            update_model.training((pre_states, updated_values))
            build(update_model)
