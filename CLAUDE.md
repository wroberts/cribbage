# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Build the Cython/C scoring extension: `pip install -e .` (config is in `pyproject.toml`; Cython is pulled in as a build requirement and compiles `cribbage/_cribbage_score.pyx` + `cribbage/cribbage_score.c` with `-std=c99`)
- Run tests: `py.test` (CI runs `py.test --cov=cribbage cribbage`)
- Single test: `py.test cribbage/test/test_scoring.py::<test_name>`

There is no lint config. The codebase is Python 3 only (`requires-python >= 3.8`; CI in `.travis.yml` runs 3.8–3.11), so use modern idioms: f-strings, `super()`, `//` for integer division. Scores must stay plain `int`s (no floats or numpy ints); `test_scoring.py` checks this for both scorers.

## Architecture

A Python library for playing cribbage and training neural-network (double Q-learning) players.

- **Game engine**: `game.py` (`Game`, `compare_players`) runs a series of `Round`s (`round.py`) between two `CribbagePlayer`s. `player.py` is the abstract interface (`discard`, `play_card`, ...); implementations are `randomplayer.py` and `simpleplayer.py`. Cards are integer values; helpers are in `cards.py`.
- **Scoring has two implementations**: pure Python `cribbage_score.py` and the C extension `_cribbage_score` (about 100x faster). Modules such as `round.py` and `neural.py` try to import from `cribbage._cribbage_score` and fall back to the Python version on `ImportError`. Keep the two behaviourally identical; `test/test_scoring.py` covers scoring. The generated `_cribbage_score.c`, `.so` and `/models` are git-ignored.
- **RL stack**:
  - `neural.py` encodes game states as vectors (e.g. `discard_state_repr`) and records player states during games.
  - `netbuilder.py` is a declarative description language for Lasagne/Theano networks, with persistent `ModelStore`/`Model` snapshots and training resumption.
  - `dqlearning.py` is the `DQLearner` double Q-learner. It uses a replay memory of (s, a, r, s2) tuples, with `None` as the terminal state.
  - `expt.py` ties these together into the training experiments. Trained models live in `models/`.
- Lasagne and Theano are old dependencies, with no requirements file.
- `discard_models.ipynb` and `priors.ipynb` are analysis notebooks, and `notes.org` holds dev notes and a TODO list.
- Ignore the `*~` backup files and `.pyc` files. They are editor and build leftovers.
