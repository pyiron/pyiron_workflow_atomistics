"""Smoke test: pyiron_workflow_atomistics.testing namespace exists."""


def test_testing_subpackage_importable():
    import pyiron_workflow_atomistics
    import pyiron_workflow_atomistics.testing

    # Public access must work via attribute (top-level __init__.py registers it)
    assert hasattr(pyiron_workflow_atomistics, "testing")


def test_engine_conformance_tests_importable():
    from pyiron_workflow_atomistics.testing import EngineConformanceTests

    assert EngineConformanceTests is not None
