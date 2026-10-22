"""structuretoolkit, pyscal3 and scikit-learn are the optional [analysis] extra.

Packages that only use, e.g., ``pyiron_workflow_atomistics.engine`` must be
able to install and import this package without them, and code that does need
them must say how to get them.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from pyiron_workflow_atomistics._internal import optional

OPTIONAL = ("structuretoolkit", "pyscal3", "sklearn")

# Every public subpackage, plus the physics topics that used to pull the
# optional packages in transitively through ``analysis/__init__``.
MODULES = (
    "pyiron_workflow_atomistics",
    "pyiron_workflow_atomistics.engine",
    "pyiron_workflow_atomistics.testing",
    "pyiron_workflow_atomistics.structure",
    "pyiron_workflow_atomistics.structure.defects",
    "pyiron_workflow_atomistics.physics",
    "pyiron_workflow_atomistics.physics.bulk",
    "pyiron_workflow_atomistics.physics.surface",
    "pyiron_workflow_atomistics.physics.point_defect",
    "pyiron_workflow_atomistics.physics.elastic",
    "pyiron_workflow_atomistics.physics.melting",
    "pyiron_workflow_atomistics.analysis",
    "pyiron_workflow_atomistics.analysis.structure_descriptors",
)


def test_core_imports_work_without_the_analysis_extra():
    """Import everything in a fresh interpreter with the extra made unimportable.

    A fresh process is needed because the optional packages may already be in
    this interpreter's ``sys.modules``; blocking them via ``sys.modules[name] =
    None`` makes any import attempt fail exactly as if they were not
    installed, whether or not they are.
    """
    script = textwrap.dedent(f"""
        import importlib, sys
        for name in {OPTIONAL!r}:
            sys.modules[name] = None
        for module in {MODULES!r}:
            importlib.import_module(module)
        loaded = [n for n in {OPTIONAL!r} if sys.modules.get(n) is not None]
        assert not loaded, loaded
        """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_core_imports_do_not_load_the_analysis_extra():
    """Even when installed, importing the package must not load the extra."""
    script = textwrap.dedent(f"""
        import importlib, sys
        for module in {MODULES!r}:
            importlib.import_module(module)
        print(",".join(n for n in {OPTIONAL!r} if n in sys.modules))
        """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


@pytest.mark.parametrize(
    ("module", "distribution"),
    [
        ("structuretoolkit.analyse", "structuretoolkit"),
        ("pyscal3", "pyscal3"),
        ("sklearn.neighbors", "scikit-learn"),
    ],
)
def test_missing_dependency_names_the_extra(monkeypatch, module, distribution):
    for name in {module.partition(".")[0], module}:
        monkeypatch.setitem(sys.modules, name, None)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        optional.require(module, "Some feature")

    message = str(excinfo.value)
    assert message.startswith("Some feature needs the optional")
    assert f"`{distribution}`" in message
    # The install command is the whole point of the guard.
    assert "pyiron_workflow_atomistics[analysis]" in message
    # The underlying failure stays reachable for debugging.
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)


def test_unrelated_missing_module_is_not_relabelled(monkeypatch):
    """A broken *installed* package keeps its own error, not the extra hint."""
    monkeypatch.setitem(sys.modules, "some_transitive_dependency", None)

    def broken_import(name):
        import some_transitive_dependency  # noqa: F401

    monkeypatch.setattr(optional.importlib, "import_module", broken_import)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        optional.require("structuretoolkit.analyse", "Some feature")

    assert "[analysis]" not in str(excinfo.value)
    assert excinfo.value.name == "some_transitive_dependency"


def test_descriptors_raise_the_actionable_error(monkeypatch):
    """Calling a public function that needs the extra names the extra."""
    from ase.build import bulk

    from pyiron_workflow_atomistics.analysis import cna_fractions

    monkeypatch.setitem(sys.modules, "structuretoolkit", None)
    monkeypatch.setitem(sys.modules, "structuretoolkit.analyse", None)

    with pytest.raises(ModuleNotFoundError, match=r"\[analysis\]"):
        cna_fractions(bulk("Al", "fcc", a=4.05, cubic=True))


def test_void_analysis_raises_the_actionable_error(monkeypatch, tmp_path):
    from ase.build import bulk

    from pyiron_workflow_atomistics.structure.defects import calculate_voids

    path = tmp_path / "cu.xyz"
    bulk("Cu", "fcc", a=3.6, cubic=True).write(path, format="extxyz")
    monkeypatch.setitem(sys.modules, "pyscal3", None)

    with pytest.raises(ModuleNotFoundError, match=r"(?s)pyscal3.*\[analysis\]"):
        calculate_voids(str(path), "extxyz", 3.6, 0.74, write=False)


@pytest.mark.parametrize("blocked", ["structuretoolkit", "sklearn"])
def test_melting_workflows_fail_before_running_the_engine(monkeypatch, blocked):
    """The melting entry points check the extra before any (expensive) engine call."""
    from pyiron_workflow_atomistics.physics.melting import (
        MeltingInput,
        calculate_melting_point,
        estimate_melting_temperature,
        melting_point_scan,
    )
    from pyiron_workflow_atomistics.physics.melting.md_steps import (
        strain_scan_nvt_nve,
    )

    class Untouchable:
        def __getattr__(self, name):
            raise AssertionError(f"engine used before the dependency check: {name}")

    engine = Untouchable()
    for name in [n for n in sys.modules if n == blocked or n.startswith(blocked + ".")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, blocked, None)

    calls = [
        lambda: calculate_melting_point(engine, MeltingInput(element="Al", a=4.05)),
        lambda: melting_point_scan(engine, MeltingInput(element="Al", a=4.05)),
        lambda: estimate_melting_temperature(None, engine, "fcc", 0.5, "fcc"),
        lambda: strain_scan_nvt_nve(None, engine, 1000.0, [1.0], "fcc"),
    ]
    for call in calls:
        with pytest.raises(ModuleNotFoundError, match=r"\[analysis\]"):
            call()
