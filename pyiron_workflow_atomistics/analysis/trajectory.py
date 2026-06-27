"""Derived quantities computed from an EngineOutput trajectory.

Engine-agnostic: works for any engine whose ``EngineOutput.structures`` carry
per-atom momenta (ASE engine; the velocity-patched LAMMPS engine) and whose
``.stresses`` hold per-frame virial stress (Voigt-6 or 3x3, in eV/A^3).
"""

from __future__ import annotations

import numpy as np
import pyiron_workflow as pwf
from ase import units

_EV_PER_A3_TO_GPA = 160.21766208


def _require_momenta(frames):
    """Raise if no frame carries momenta (engine failed to record velocities)."""
    if all(np.abs(np.asarray(f.get_momenta())).sum() == 0.0 for f in frames):
        raise ValueError(
            "Trajectory frames carry no momenta; kinetic temperature would be a "
            "silent 0 K. The engine must record per-atom velocities for MD analyses."
        )


def _virial_pressure_ev_per_a3(stress: np.ndarray) -> float:
    """Hydrostatic virial pressure P = -tr(sigma)/3 from Voigt-6 or 3x3 stress."""
    s = np.asarray(stress, dtype=float)
    if s.shape == (6,):
        trace = s[0] + s[1] + s[2]
    elif s.shape == (3, 3):
        trace = s[0, 0] + s[1, 1] + s[2, 2]
    else:
        raise ValueError(f"Unexpected stress shape {s.shape}; expected (6,) or (3, 3)")
    return -trace / 3.0


@pwf.as_function_node("temperature")
def temperatures_from_trajectory(engine_output, last_n: int = 20) -> float:
    """Mean kinetic temperature (K) over the last ``last_n`` trajectory frames."""
    frames = engine_output.structures
    if not frames:
        raise ValueError("engine_output.structures is empty; need an MD trajectory")
    window = frames[-last_n:]
    _require_momenta(window)
    temperature = float(np.mean([f.get_temperature() for f in window]))
    return temperature


@pwf.as_function_node("pressure")
def pressures_from_trajectory(engine_output, last_n: int = 20) -> float:
    """Mean total pressure (GPa) over the last ``last_n`` frames: virial + kinetic.

    Assumes ``EngineOutput.stresses`` are the **potential/virial stress in eV/Å³
    with the ASE sign convention** (P = -tr(sigma)/3), as the ASEEngine emits; the
    kinetic ideal-gas term N·kB·T/V is added here. NOTE: the LAMMPS engine instead
    emits a *total* pressure tensor in GPa (compression-positive), so this node is
    not yet correct for LAMMPS output — reconciling the stress convention across
    engines is a prerequisite for the LAMMPS coexistence pressure (see the
    verification report).
    """
    frames = engine_output.structures
    stresses = engine_output.stresses
    if not frames or stresses is None or len(stresses) == 0:
        raise ValueError("engine_output needs both .structures and .stresses")
    window_f = frames[-last_n:]
    window_s = stresses[-last_n:]
    _require_momenta(window_f)
    pressures = []
    for frame, stress in zip(window_f, window_s):
        p_vir = _virial_pressure_ev_per_a3(stress)  # eV/A^3
        p_kin = (
            len(frame) * units.kB * frame.get_temperature() / frame.get_volume()
        )  # eV/A^3
        pressures.append((p_vir + p_kin) * _EV_PER_A3_TO_GPA)  # GPa
    return float(np.mean(pressures))
