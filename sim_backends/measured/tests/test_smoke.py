"""M0 import smoke tests for the measured backend package."""


def test_import_measured():
    import sim_backends.measured  # noqa: F401
    import sim_backends.measured.campaign  # noqa: F401
    import sim_backends.measured.oracles  # noqa: F401


def test_existing_registry_unaffected():
    # measured is not registered until M5; legacy/upstream stay intact.
    import sim_backends

    assert set(sim_backends.list_backends()) >= {"legacy", "upstream"}
