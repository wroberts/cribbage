#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
utils.py
(c) Will Roberts  11 January, 2017

Utility functions.
'''

import itertools

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    iter1, iter2 = itertools.tee(iterable)
    next(iter2, None)
    return itertools.izip(iter1, iter2)

def doubler(iterable):
    '''(x, y, z, ...) -> ((x, x), (y, y), (z, z), ...)'''
    for val in iterable:
        yield (val, val)
