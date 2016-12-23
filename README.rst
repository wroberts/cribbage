There is a C function which can score a cribbage hand 100 times faster
than doing it in pure Python.

To compile the Cython extension::

    python setup.py build_ext --inplace

Then::

    import c_cribbage_score
    c_cribbage_score.score_hand(hand, draw)
