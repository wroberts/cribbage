==========
 Cribbage
==========

Python library for experimenting with the game of cribbage.

C Extension
===========

There is a C function which can score a cribbage hand 100 times faster
than doing it in pure Python.

To compile the Cython extension::

    python setup.py build_ext --inplace

Then::

    import c_cribbage_score
    c_cribbage_score.score_hand(hand, draw)

Testing
=======

In the base directory of the project, do::

    py.test

References
==========

1. Mnih, Volodymyr, et al. `Human-level control through deep
   reinforcement learning`_. Nature 518, no. 7540 (2015): 529-533.
2. van Hasselt, Hado. `Double Q-learning`_. Poster at Advances in
   Neural Information Processing Systems 23 (NIPS 2010), Vancouver,
   British Columbia, Canada.

.. _`Human-level control through deep reinforcement learning`: http://www.davidqiu.com:8888/research/nature14236.pdf
.. _`Double Q-learning`: https://hadovanhasselt.files.wordpress.com/2015/12/doubleqposter.pdf
