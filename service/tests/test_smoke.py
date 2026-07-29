"""M0 import smoke tests for the service package."""


def test_import_service():
    import service  # noqa: F401


def test_import_subpackages():
    import service.api  # noqa: F401
    import service.inventory  # noqa: F401
