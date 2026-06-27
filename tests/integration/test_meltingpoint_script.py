import importlib.util
import pathlib

import pytest


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "meltingpoint", pathlib.Path("scripts/meltingpoint.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
def test_script_run_returns_result(tmp_path):
    mod = _load_script()
    res = mod.run(
        element="Al",
        a=4.05,
        n_atoms=500,
        working_directory=str(tmp_path),
        full=False,
        temperature_right=1400.0,
        strain_run_steps=40,
        seed=1,
    )
    assert res["initial_guess"] >= 0
