"""Tests for the offline research lab.

Run from the repository root with either:

    python3 -m unittest discover -s tests -t .
    python3 -m pytest tests

Scope: these cover ``research/`` only. The forecaster in ``main.py`` is
deliberately untested here -- this milestone builds the instrument, and adding
tests that pin production behaviour would invite changing production behaviour
to satisfy them.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
