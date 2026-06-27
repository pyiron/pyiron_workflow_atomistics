#!/usr/bin/env python
"""Interface-method melting point - runnable port of meltingpoint.ipynb.

Step 1 (rough estimate) by default; pass --full for the full coexistence method.
Uses the ASE engine with EMT by default (swap in a GRACE calculator for production).

Examples
--------
    python scripts/meltingpoint.py --element Al --a 4.05            # Step-1 estimate
    python scripts/meltingpoint.py --element Al --a 4.05 --full     # full method
"""

from __future__ import annotations

import argparse


def run(
    element="Al",
    crystalstructure="fcc",
    a=4.05,
    n_atoms=4000,
    working_directory="melting_run",
    full=False,
    temperature_right=1000.0,
    strain_run_steps=1000,
    seed=12345,
):
    from ase.calculators.emt import EMT

    from pyiron_workflow_atomistics.analysis.structure_descriptors import (
        analyse_reference_structure,
    )
    from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
    from pyiron_workflow_atomistics.physics.melting import (
        MeltingInput,
        calculate_melting_point,
    )
    from pyiron_workflow_atomistics.physics.melting.initial_guess import (
        estimate_melting_temperature,
    )
    from pyiron_workflow_atomistics.physics.melting.structures import (
        create_coexistence_supercell,
    )

    engine = ASEEngine(
        EngineInput=CalcInputStatic(), calculator=EMT(),
        working_directory=working_directory,
    )
    if full:
        mi = MeltingInput(
            element=element, crystalstructure=crystalstructure, a=a, n_atoms=n_atoms,
            temperature_right=temperature_right, strain_run_steps=strain_run_steps,
            seed=seed,
        )
        res = calculate_melting_point.node_function(engine, mi).to_dict()
        print(
            f"Melting temperature: {res['melting_temperature']:.1f} K "
            f"(converged={res['converged']}, guess={res['initial_guess']:.0f} K)"
        )
        return res

    structure = create_coexistence_supercell.node_function(
        element, crystalstructure, a=a, n_atoms=n_atoms
    )
    key_max, _, half = analyse_reference_structure.node_function(structure)
    t_guess, _ = estimate_melting_temperature.node_function(
        structure, engine, key_max=key_max, distribution_half=half,
        crystalstructure=crystalstructure, temperature_right=temperature_right,
        strain_run_steps=strain_run_steps, seed=seed,
    )
    print(f"Step-1 melting-temperature estimate: {t_guess} K")
    return {"initial_guess": float(t_guess)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--element", default="Al")
    p.add_argument("--crystalstructure", default="fcc")
    p.add_argument("--a", type=float, default=4.05)
    p.add_argument("--n-atoms", type=int, default=4000)
    p.add_argument("--working-directory", default="melting_run")
    p.add_argument("--full", action="store_true")
    p.add_argument("--temperature-right", type=float, default=1000.0)
    p.add_argument("--strain-run-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    run(
        element=args.element, crystalstructure=args.crystalstructure, a=args.a,
        n_atoms=args.n_atoms, working_directory=args.working_directory, full=args.full,
        temperature_right=args.temperature_right,
        strain_run_steps=args.strain_run_steps, seed=args.seed,
    )


if __name__ == "__main__":
    main()
