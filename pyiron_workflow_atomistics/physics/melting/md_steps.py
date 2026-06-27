from __future__ import annotations

from dataclasses import replace

import pyiron_workflow as pwf

from pyiron_workflow_atomistics.analysis.structure_descriptors import voronoi_max_mean
from pyiron_workflow_atomistics.analysis.trajectory import (
    pressures_from_trajectory,
    temperatures_from_trajectory,
)
from pyiron_workflow_atomistics.engine import CalcInputMD, calculate
from pyiron_workflow_atomistics.physics.melting.solid_fraction import solid_fraction_kde
from pyiron_workflow_atomistics.physics.melting.structures import (
    freeze_half,
    strain_cell_along_z,
    unfreeze,
)


def _engine_with(engine, calc_input, subdir):
    """Engine copy carrying ``calc_input`` and a per-step working subdirectory."""
    return replace(engine, EngineInput=calc_input).with_working_directory(subdir)


@pwf.as_function_node("relaxed_structure", "engine_output")
def npt_relax_solid(
    structure,
    engine,
    temperature,
    n_steps=10000,
    timestep=2.0,
    seed=None,
    npt_thermostat="berendsen",
    subdir="npt_solid",
):
    """NPT MD on the bulk solid at ``temperature`` (P=0).

    ``npt_thermostat`` must yield an *isotropic* (orthorhombic) cell so the
    downstream CNA/Voronoi analyses stay valid: use ``"berendsen"`` for the ASE
    engine (scalar-pressure, orthorhombic) and ``"nose-hoover"`` for the LAMMPS
    engine (``fix npt ... iso``). Full triclinic NPT must be avoided.
    """
    md = CalcInputMD(
        mode="NPT",
        thermostat=npt_thermostat,
        temperature=temperature,
        pressure=0.0,
        n_ionic_steps=n_steps,
        n_print=max(1, n_steps // 100),
        time_step=timestep,
        initial_temperature=2.0 * temperature,
        seed=seed,
        compressibility=1e-6,
    )
    out = calculate.node_function(structure, engine=_engine_with(engine, md, subdir))
    return out.final_structure, out


@pwf.as_function_node("interface_structure")
def build_solid_liquid_interface(
    structure,
    engine,
    t_solid,
    t_liquid,
    n_steps=10000,
    timestep=2.0,
    seed=None,
    subdir="interface",
):
    """Freeze lower half, melt upper half at t_liquid (NVT), recool to t_solid."""
    frozen = freeze_half.node_function(structure)
    melt_md = CalcInputMD(
        mode="NVT",
        thermostat="langevin",
        temperature=t_liquid,
        n_ionic_steps=n_steps,
        n_print=max(1, n_steps // 100),
        time_step=timestep,
        initial_temperature=2.0 * t_liquid,
        seed=seed,
    )
    melted = calculate.node_function(
        frozen, engine=_engine_with(engine, melt_md, f"{subdir}_melt")
    ).final_structure
    cool_md = CalcInputMD(
        mode="NVT",
        thermostat="langevin",
        temperature=t_solid,
        n_ionic_steps=n_steps,
        n_print=max(1, n_steps // 100),
        time_step=timestep,
        initial_temperature=2.0 * t_solid,
        seed=seed,
    )
    cooled = calculate.node_function(
        melted, engine=_engine_with(engine, cool_md, f"{subdir}_cool")
    ).final_structure
    interface_structure = unfreeze.node_function(cooled)
    return interface_structure


@pwf.as_function_node("records")
def strain_scan_nvt_nve(
    structure,
    engine,
    temperature,
    strains,
    crystalstructure,
    nvt_steps=10000,
    nve_steps=20000,
    timestep=2.0,
    seed=None,
    last_n=20,
    subdir="strain",
):
    """For each strain: NVT-equilibrate then NVE; record T, P, solid fraction, voronoi."""
    records = []
    for i, strain in enumerate(strains):
        strained = strain_cell_along_z.node_function(structure, strain)
        nvt_md = CalcInputMD(
            mode="NVT",
            thermostat="langevin",
            temperature=temperature,
            n_ionic_steps=nvt_steps,
            n_print=max(1, nvt_steps // 100),
            time_step=timestep,
            initial_temperature=2.0 * temperature,
            seed=seed,
        )
        equil = calculate.node_function(
            strained, engine=_engine_with(engine, nvt_md, f"{subdir}_nvt_{i:03d}")
        ).final_structure
        # The NVE input `equil` is already NVT-equilibrated at `temperature`
        # (thermally warm, PE above the lattice minimum). The 2*T half-velocity
        # trick is only valid from a COLD lattice; re-seeding a warm config at 2*T
        # equipartitions to a kinetic temperature ~1.5*T. Re-seed at 1*T so the
        # thermostat-free NVE settles back to ~T (the measured coexistence T).
        nve_md = CalcInputMD(
            mode="NVE",
            temperature=temperature,
            n_ionic_steps=nve_steps,
            n_print=max(1, nve_steps // 100),
            time_step=timestep,
            initial_temperature=temperature,
            seed=seed,
        )
        nve_out = calculate.node_function(
            equil, engine=_engine_with(engine, nve_md, f"{subdir}_nve_{i:03d}")
        )
        vmax, vmean = voronoi_max_mean.node_function(nve_out.final_structure)
        records.append(
            {
                "strain": strain,
                "mean_T": temperatures_from_trajectory.node_function(
                    nve_out, last_n=last_n
                ),
                "mean_P": pressures_from_trajectory.node_function(nve_out, last_n=last_n),
                "solid_fraction": solid_fraction_kde.node_function(
                    nve_out.final_structure, crystalstructure
                ),
                "voronoi_max": vmax,
                "voronoi_mean": vmean,
            }
        )
    return records
