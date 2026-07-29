"""Root pytest conftest: default-mark unmarked tests as ``unit``.

pytest.ini sets ``addopts = -m unit``; without this hook every pre-existing
(unmarked) test would be silently deselected. Tests opt out of the default by
carrying ``@pytest.mark.sim`` or ``@pytest.mark.hw``.
"""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.get_closest_marker("sim") is None and item.get_closest_marker("hw") is None:
            item.add_marker(pytest.mark.unit)
