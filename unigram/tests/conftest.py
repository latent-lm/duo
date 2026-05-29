"""Pytest bootstrap for the ``unigram`` geo/heat-kernel tests.

There is no package ``__init__.py`` under ``unigram/`` and no installed
distribution, so put the ``unigram/`` directory itself on ``sys.path`` here.
That makes the flat ``import geo_bridge`` (and ``import hyper_bridge``) resolve
exactly the way the modules import each other internally.
"""

from __future__ import annotations

import os
import sys

# ``unigram/`` is the parent of this ``tests/`` directory.
_UNIGRAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _UNIGRAM_DIR not in sys.path:
    sys.path.insert(0, _UNIGRAM_DIR)
