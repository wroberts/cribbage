#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
utils.py
(c) Will Roberts  11 January, 2017

Utility functions.
'''

from contextlib import contextmanager
from functools import wraps
import errno
import itertools
import os
import random
import tempfile
import numpy as np

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    iter1, iter2 = itertools.tee(iterable)
    next(iter2, None)
    return itertools.izip(iter1, iter2)

def doubled(iterable):
    '''(x, y, z, ...) -> ((x, x), (y, y), (z, z), ...)'''
    for val in iterable:
        yield (val, val)

# http://stackoverflow.com/a/8991553/1062499
def grouped(num, iterable):
    '''
    >>> list(grouped(3, 'ABCDEFG'))
    [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)]

    Arguments:
    - `n`:
    - `iterable`:
    '''
    iterator = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(iterator, num))
        if not chunk:
            return
        yield chunk

def random_skip(seq, prob=0.2):
    '''
    Generator yields items from a sequence, randomly skipping some of
    them.

    Arguments:
    - `seq`:
    - `prob`: the probability of emitting an item
    '''
    for item in seq:
        if random.random() < prob:
            yield item

def mkdir_p(path):
    '''
    Functionality similar to mkdir -p.
    '''
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def numpy_memoize(filename):
    '''
    Function decorator to cache the results of a function call in a
    numpy archive.
    '''
    def numpy_memoize_decorator(func):
        '''Inner function produced by numpy_memoize.'''
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            '''Function wrapper used by numpy_memoize.'''
            try:
                with np.load(filename) as input_file:
                    values = [input_file['arr_%d' % i] for i in
                              range(len(input_file.files))]
            except IOError:
                values = func(*args, **kwargs)
                np.savez(filename, *values)
            return values
        return func_wrapper
    return numpy_memoize_decorator

# ------------------------------------------------------------
#  Atomic file I/O
# ------------------------------------------------------------
# http://stackoverflow.com/a/29491523/1062499

@contextmanager
def tempfilename(suffix='', directory=None):
    '''
    Context manager for creating temporary file names.

    Will find a free temporary filename upon entering and will try to
    delete the file on leaving, even in case of an exception.

    Arguments:
    - `suffix`: string, optional file suffix
    - `directory`: string, optional directory to save temporary file in
    '''
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=directory)
    tfile.file.close()
    try:
        yield tfile.name
    finally:
        try:
            os.remove(tfile.name)
        except OSError as err:
            if err.errno == errno.ENOENT: # No such file or directory
                pass
            else:
                raise

@contextmanager
def open_atomic(filepath, mode='w+b', fsync=False, **kwargs):
    '''
    Open temporary file object that atomically moves to destination
    upon exiting.

    Allows reading and writing to and from the same filename.

    The file will not be moved to destination in case of an exception.

    Arguments:
    - `filepath`: string, the file path to be opened
    - `fsync`: bool, whether to force write the file to disk
    - `**kwargs`: mixed, Any valid keyword arguments for `open`
    '''
    with tempfilename(directory=os.path.dirname(os.path.abspath(filepath))) as tmppath:
        with open(tmppath, mode, **kwargs) as output_file:
            try:
                yield output_file
            finally:
                if fsync:
                    output_file.flush()
                    os.fsync(output_file.fileno())
        os.rename(tmppath, filepath)
