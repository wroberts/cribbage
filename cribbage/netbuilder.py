#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
netbuilder.py
(c) Will Roberts  11 January, 2017

Declarative description language for producing persistent neural
network models.
'''

import itertools

def doubler(iterable):
    '''(x, y, z, ...) -> ((x, x), (y, y), (z, z), ...)'''
    for x in iterable:
        yield (x, x)

# models will be stored in the models/ directory
store = ModelStore('models')

# create and configure a new model
dautoenc = Model(store, 'dautoenc')
# network architecture
dautoenc.input(294)
dautoenc.hidden(150, 'rectify') # Dense
dautoenc.output(294, 'rectify') # Dense
dautoenc.objective('squared_error')
dautoenc.update('adadelta')
# build a validation set with fixed random state
val_set = itertools.islice(doubler(random_discard_state_gen(42)), 500)
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
dautoenc2.input(294)
dautoenc2.hidden(150, 'rectify') # Dense
dautoenc2.hidden(150, 'rectify') # Dense
dautoenc2.output(294, 'rectify') # Dense
dautoenc2.objective('squared_error')
dautoenc2.update('adadelta')
# initialise weights on first layer
dautoenc = Model(store, 'dautoenc').best_validation_error
dautoenc2.set_weights('hidden1', dautoenc.get_weights('hidden1'))
# build a validation set with fixed random state
val_set = itertools.islice(doubler(random_discard_state_gen(42)), 500)
dautoenc2.validation(val_set)
# training stream with non-fixed random state
stream = doubler(random_discard_state_gen())
dautoenc2.training(stream)
# configure training loop
dautoenc2.minibatch_size(500)
dautoenc2.num_minibatches(10000)
# build the model
build(dautoenc2)
