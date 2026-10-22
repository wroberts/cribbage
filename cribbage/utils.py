#!/usr/bin/env python

"""
utils.py
(c) Will Roberts  11 January, 2017

Utility functions.
"""

from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from functools import wraps
import errno
import itertools
import os
import random
import tempfile
from typing import IO, Any, TypeVar
import numpy as np

T = TypeVar("T")


def pairwise(iterable: Iterable[T]) -> Iterator[tuple[T, T]]:
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    iter1, iter2 = itertools.tee(iterable)
    next(iter2, None)
    return zip(iter1, iter2)


def doubled(iterable: Iterable[T]) -> Iterator[tuple[T, T]]:
    """(x, y, z, ...) -> ((x, x), (y, y), (z, z), ...)"""
    for val in iterable:
        yield (val, val)


# http://stackoverflow.com/a/8991553/1062499
def grouped(num: int, iterable: Iterable[T]) -> Iterator[tuple[T, ...]]:
    """
    >>> list(grouped(3, 'ABCDEFG'))
    [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)]

    Arguments:
    - `n`:
    - `iterable`:
    """
    iterator = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(iterator, num))
        if not chunk:
            return
        yield chunk


def random_skip(seq: Iterable[T], prob: float = 0.2) -> Iterator[T]:
    """
    Generator yields items from a sequence, randomly skipping some of
    them.

    Arguments:
    - `seq`:
    - `prob`: the probability of emitting an item
    """
    for item in seq:
        if random.random() < prob:
            yield item


def mkdir_p(path: str | os.PathLike[str]) -> None:
    """
    Functionality similar to mkdir -p.
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def numpy_memoize(
    filename: str,
) -> Callable[[Callable[..., Sequence[Any]]], Callable[..., Sequence[Any]]]:
    """
    Function decorator to cache the results of a function call in a
    numpy archive.
    """

    def numpy_memoize_decorator(
        func: Callable[..., Sequence[Any]],
    ) -> Callable[..., Sequence[Any]]:
        """Inner function produced by numpy_memoize."""

        @wraps(func)
        def func_wrapper(*args: Any, **kwargs: Any) -> Sequence[Any]:
            """Function wrapper used by numpy_memoize."""
            values: Sequence[Any]
            try:
                with np.load(filename) as input_file:
                    values = [
                        input_file[f"arr_{i}"] for i in range(len(input_file.files))
                    ]
            except OSError:
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
def tempfilename(
    suffix: str = "", directory: str | None = None
) -> Generator[str, None, None]:
    """
    Context manager for creating temporary file names.

    Will find a free temporary filename upon entering and will try to
    delete the file on leaving, even in case of an exception.

    Arguments:
    - `suffix`: string, optional file suffix
    - `directory`: string, optional directory to save temporary file in
    """
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=directory)
    tfile.file.close()
    try:
        yield tfile.name
    finally:
        try:
            os.remove(tfile.name)
        except OSError as err:
            if err.errno == errno.ENOENT:  # No such file or directory
                pass
            else:
                raise


@contextmanager
def open_atomic(
    filepath: str,
    mode: str = "w+b",
    fsync: bool = False,
    **kwargs: Any,
) -> Generator[IO[Any], None, None]:
    """
    Open temporary file object that atomically moves to destination
    upon exiting.

    Allows reading and writing to and from the same filename.

    The file will not be moved to destination in case of an exception.

    Arguments:
    - `filepath`: string, the file path to be opened
    - `fsync`: bool, whether to force write the file to disk
    - `**kwargs`: mixed, Any valid keyword arguments for `open`
    """
    with tempfilename(directory=os.path.dirname(os.path.abspath(filepath))) as tmppath:
        with open(tmppath, mode, **kwargs) as output_file:
            try:
                yield output_file
            finally:
                if fsync:
                    output_file.flush()
                    os.fsync(output_file.fileno())
        os.rename(tmppath, filepath)
