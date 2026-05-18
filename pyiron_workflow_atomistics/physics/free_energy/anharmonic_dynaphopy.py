"""Anharmonic free energy via dynaphopy MD-projection — single T and TDI over T."""

from __future__ import annotations

import numpy as np
import pyiron_workflow as pwf
from ase import Atoms

from pyiron_workflow_atomistics.engine import Engine
from pyiron_workflow_atomistics.physics.free_energy.harmonic import _resolve_simfolder
from pyiron_workflow_atomistics.physics.free_energy.outputs import FreeEnergyOutput
from pyiron_workflow_atomistics.physics.phonons._compat import (
    ir_qpoints_and_weights,
    require_phonopy,
)
from pyiron_workflow_atomistics.physics.phonons.md_renormalised import (
    calculate_phonon_md_renormalisation,
)


@pwf.as_function_node("free_energy_per_atom", "entropy_per_atom", "cv_per_atom")
def _free_energy_from_spectrum(
    frequencies: np.ndarray,  # (n_q, n_band) THz
    q_weights: np.ndarray,  # (n_q,), sums to 1
    temperature: float,  # K
    n_atoms_primitive: int,
) -> tuple[float, float, float]:
    """Harmonic free energy / entropy / Cv on a discrete (q, band) frequency grid.

    F = sum_q w_q * sum_b [ ℏω_qb/2 + k_B T ln(1 − exp(−ℏω_qb / k_B T)) ]

    Acoustic modes at Γ are zeroed upstream by dynaphopy's
    `_project_with_dynaphopy` (positions[:3]=0 at q==0); any ω ≤ 0 remaining
    is treated as imaginary and rejected.

    Units
    -----
    Frequencies in THz. Returned F / S / Cv per primitive-cell atom, in
    eV / (eV/K) / (eV/K) respectively.
    """
    import scipy.constants as c

    freqs = np.asarray(frequencies, dtype=float)
    weights = np.asarray(q_weights, dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"q_weights must sum to 1, got {weights.sum()}")
    if freqs.ndim != 2 or freqs.shape[0] != weights.shape[0]:
        raise ValueError(
            f"frequencies shape {freqs.shape} incompatible with q_weights shape "
            f"{weights.shape}"
        )

    # Acoustic-at-Γ modes (already zeroed upstream) are dropped from the sum.
    is_acoustic_gamma = freqs == 0.0
    active = freqs[~is_acoustic_gamma]
    imag_mask = active < 0
    if imag_mask.any():
        raise ValueError(
            f"Spectrum has {int(imag_mask.sum())} imaginary modes; "
            "harmonic free energy is undefined for an unstable spectrum."
        )

    omega_rad_s = freqs * 1e12 * 2 * np.pi
    hbar_omega_eV = c.hbar * omega_rad_s / c.eV  # (n_q, n_band)

    if temperature <= 0:
        # T=0: only zero-point energy contributes; S and Cv are zero.
        F_modes = 0.5 * hbar_omega_eV
        F_modes = np.where(is_acoustic_gamma, 0.0, F_modes)
        F = float((weights[:, None] * F_modes).sum() / n_atoms_primitive)
        S = 0.0
        Cv = 0.0
    else:
        kT_eV = c.Boltzmann * temperature / c.eV
        x = hbar_omega_eV / kT_eV
        # ln(1 − exp(−x)) — stable; modes with x huge → ln(1) = 0.
        log_term = np.where(is_acoustic_gamma, 0.0, np.log1p(-np.exp(-x)))
        F_modes = 0.5 * hbar_omega_eV + kT_eV * log_term
        F_modes = np.where(is_acoustic_gamma, 0.0, F_modes)

        # Entropy per mode:  S = k_B [ x/(exp(x)−1) − ln(1 − exp(−x)) ]
        kB_eV_per_K = c.Boltzmann / c.eV
        with np.errstate(over="ignore", invalid="ignore"):
            expm1_x = np.expm1(x)
            # Guard expm1 underflow for ultra-soft modes (same condition Cv uses).
            bose_term = np.where(is_acoustic_gamma | (expm1_x == 0), 0.0, x / expm1_x)
            S_modes = kB_eV_per_K * (bose_term - log_term)

        # Cv per mode:  C_v = k_B (x^2 exp(x)) / (exp(x)−1)^2
        with np.errstate(over="ignore", invalid="ignore"):
            denom = expm1_x**2
            num = (x**2) * np.exp(x)
            Cv_modes = kB_eV_per_K * np.where(
                is_acoustic_gamma | (denom == 0), 0.0, num / denom
            )

        F = float((weights[:, None] * F_modes).sum() / n_atoms_primitive)
        S = float((weights[:, None] * S_modes).sum() / n_atoms_primitive)
        Cv = float((weights[:, None] * Cv_modes).sum() / n_atoms_primitive)

    return F, S, Cv


def _commensurate_q_points(structure: Atoms, q_mesh) -> tuple[np.ndarray, np.ndarray]:
    """Return (q_points, weights) on a commensurate Monkhorst-Pack mesh.

    The mesh is laid out on the *primitive* cell that phonopy infers via
    primitive_matrix="auto", so it stays consistent with the FC2 view
    dynaphopy projects into. Weights sum to 1.

    The actual ir-grid reduction is delegated to
    :func:`pyiron_workflow_atomistics.physics.phonons._compat.ir_qpoints_and_weights`,
    which bridges the phonopy v3/v4 API split (``GridPoints`` → ``get_ir_qpoints_and_weights``).
    """
    require_phonopy()
    import phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    unitcell = PhonopyAtoms(
        symbols=list(structure.get_chemical_symbols()),
        positions=structure.get_positions(),
        cell=np.asarray(structure.get_cell()),
        masses=structure.get_masses(),
    )
    phonon = phonopy.Phonopy(
        unitcell=unitcell,
        supercell_matrix=np.eye(3, dtype=int),
        primitive_matrix="auto",
    )
    q_points, weights = ir_qpoints_and_weights(
        mesh=q_mesh,
        phonopy_obj=phonon,
        is_gamma_center=True,
        is_mesh_symmetry=True,
    )
    weights = weights / weights.sum()
    return q_points, weights


@pwf.as_function_node("q_points", "q_weights")
def _commensurate_q_points_node(structure: Atoms, q_mesh):
    """Function-node wrapper for `_commensurate_q_points`.

    Runs the phonopy mesh resolution at execution time rather than at graph
    construction time, so the macro body never touches proxy ``UserInput``
    arguments when computing the q-mesh.
    """
    q_points, q_weights = _commensurate_q_points(structure, q_mesh)
    return q_points, q_weights


@pwf.as_function_node("n_atoms")
def _n_atoms_node(structure: Atoms) -> int:
    """Return ``len(structure)`` at execution time.

    Function-node wrapper so the macro body never calls ``len`` on a proxy
    ``UserInput`` object — pyiron_workflow proxies don't support ``len``.
    """
    return len(structure)


@pwf.as_function_node("guarded_frequencies", "n_guarded")
def _guard_unphysical_frequencies(
    renormalised_frequencies: np.ndarray,
    harmonic_frequencies: np.ndarray,
    max_relative_shift: float = 0.5,
):
    """Fall back to harmonic frequencies for clearly-unphysical renorm fits.

    dynaphopy fits each (q, band) Lorentzian independently. With short MD
    trajectories or noisy power spectra, the fit can produce peak positions
    far from the harmonic value — even at >2x the true frequency — which
    would inflate the harmonic-formula free energy spuriously.

    Any band whose renormalised frequency differs from its harmonic value by
    more than ``max_relative_shift`` (default 50%) is replaced by the
    harmonic value. NaN renormalised entries are also replaced. The check
    runs per-(q, band) so good fits are preserved.

    Returns both the guarded array and the count of substituted entries so
    callers can warn / record how much of the result fell back to harmonic.
    """
    renorm = np.asarray(renormalised_frequencies, dtype=float)
    harm = np.asarray(harmonic_frequencies, dtype=float)
    if renorm.shape != harm.shape:
        raise ValueError(
            f"renormalised shape {renorm.shape} != harmonic shape {harm.shape}"
        )

    out = renorm.copy()
    # Clamp tiny numerical noise around the acoustic-at-Γ branches (phonopy
    # leaves these as ~1e-6 THz of either sign). dynaphopy already pins
    # `renormalised_frequencies[:3]` to 0 at q==0; mirror that on the
    # harmonic side so the fallback path does not reintroduce sub-µeV
    # imaginary modes that would trip `_free_energy_from_spectrum`.
    near_zero_atol = 1e-3  # THz; well below any physical phonon band.
    harm = np.where(np.abs(harm) < near_zero_atol, 0.0, harm)

    abs_harm = np.abs(harm)
    relative_shift = np.where(
        abs_harm > 0,
        np.abs(renorm - harm) / np.where(abs_harm > 0, abs_harm, 1.0),
        0.0,
    )
    needs_guard = ~np.isfinite(renorm) | (relative_shift > max_relative_shift)
    out = np.where(needs_guard, harm, out)
    n_guarded = int(np.sum(needs_guard))
    return out, n_guarded


@pwf.as_function_node("free_energy_output")
def _pack_anharmonic_dynaphopy_output(
    structure: Atoms,
    md_phonon_output,
    q_weights: np.ndarray,
    free_energy_per_atom: float,
    entropy_per_atom: float,
    cv_per_atom: float,
    temperature: float,
    q_mesh,
    n_guarded: int,
    simfolder: str,
    keep_handles: bool,
) -> FreeEnergyOutput:
    """Pack F/S/Cv (eV/atom; (eV/K)/atom) plus dynaphopy outputs into FreeEnergyOutput.

    Units follow the ``FreeEnergyOutput`` spec: ``free_energy`` in eV/atom,
    ``entropy`` / ``heat_capacity`` in (eV/K)/atom. ``_free_energy_from_spectrum``
    already produces values in these units, so no conversion is needed here.

    Records the count of (q, band) entries that fell back to harmonic via
    ``_guard_unphysical_frequencies`` in ``report["n_guarded_modes"]`` and
    emits a ``UserWarning`` when any guard fired — the renormalised spectrum
    was not entirely trusted and the result is a partial harmonic fall-back.
    """
    import warnings

    healthy, issues = md_phonon_output.check_md_health()
    elements = list(dict.fromkeys(structure.get_chemical_symbols()))
    q_mesh_tuple = tuple(int(x) for x in q_mesh)
    if n_guarded > 0:
        n_total = int(
            np.prod(np.asarray(md_phonon_output.renormalised_frequencies).shape)
        )
        warnings.warn(
            f"_guard_unphysical_frequencies replaced {n_guarded}/{n_total} (q, band) "
            "entries with their harmonic values (renormalisation differed by >50% or was "
            "NaN). The result reflects a partial fall-back to harmonic; bump "
            "`production_steps` or `q_mesh` for more trustworthy anharmonic values.",
            stacklevel=2,
        )
    return FreeEnergyOutput(
        mode="anharmonic_dynaphopy",
        reference_phase="solid",
        free_energy=float(free_energy_per_atom),
        free_energy_error=0.0,
        temperature=float(temperature),
        pressure=0.0,
        n_atoms=len(structure),
        elements=elements,
        simfolder=simfolder,
        report={
            "method": "anharmonic_dynaphopy",
            "n_md_steps": int(md_phonon_output.n_md_steps),
            "time_step_fs": float(md_phonon_output.time_step_fs),
            "md_temperature_mean": float(md_phonon_output.md_temperature_mean),
            "md_temperature_std": float(md_phonon_output.md_temperature_std),
            "md_health": {"healthy": bool(healthy), "issues": list(issues)},
            "n_guarded_modes": int(n_guarded),
        },
        entropy=float(entropy_per_atom),
        heat_capacity=float(cv_per_atom),
        harmonic_frequencies=np.asarray(md_phonon_output.harmonic_frequencies),
        renormalised_frequencies=np.asarray(md_phonon_output.renormalised_frequencies),
        linewidths=np.asarray(md_phonon_output.linewidths),
        q_mesh=q_mesh_tuple,
        dynaphopy_handle=(md_phonon_output.quasiparticle if keep_handles else None),
        phonopy_handle=md_phonon_output.phonopy if keep_handles else None,
    )


@pwf.api.as_macro_node("free_energy_output")
def anharmonic_free_energy_dynaphopy(
    wf,
    structure: Atoms,
    engine: Engine,
    fc2_supercell_matrix,
    temperature: float,
    equilibration_steps: int = 2000,
    production_steps: int = 10000,
    time_step: float = 1.0,
    thermostat_time_constant: float = 100.0,
    seed: int | None = None,
    q_mesh=(11, 11, 11),
    phono3py_output=None,
    working_directory: str = ".",
    subdir: str = "anharmonic_free_energy_dynaphopy",
    keep_handles: bool = False,
):
    """Anharmonic free energy at one T via dynaphopy MD projection + harmonic-formula sum.

    This macro applies a defensive guard: any (q, band) whose renormalised
    frequency deviates from its harmonic counterpart by more than 50% (or is
    NaN) is replaced by the harmonic value, and the number of guarded entries
    is recorded in ``report["n_guarded_modes"]`` with a warning when it fires.
    The default 50% threshold catches under-converged Lorentzian fits from
    short MD trajectories; bump ``production_steps`` and ``q_mesh`` for results
    that lean less on the harmonic fall-back.

    Engine must expose ``.calculator`` (inherited from
    ``calculate_phonon_md_renormalisation``).

    Notes
    -----
    The macro passes ``n_atoms_primitive=len(structure)`` to
    ``_free_energy_from_spectrum`` so the returned F/S/Cv are per *unit-cell*
    atom — i.e., per atom of the user-supplied ``structure``. This is correct
    when ``structure`` is itself a primitive cell, but it is also the unit
    that callers typically expect when ``structure`` is a conventional cell
    (e.g. ``bulk("Al", "fcc", cubic=True)`` with 4 atoms): dynaphopy returns
    ``renormalised_frequencies`` with shape ``(n_q, 3 * n_atoms_unitcell)``,
    so summing over all bands and dividing by ``len(structure)`` gives a
    per-atom F that matches the per-atom F produced upstream by
    ``_compute_harmonic_observables``. For genuinely non-primitive non-cubic
    inputs the result will still be per-unit-cell-atom, which is the consistent
    convention across the free-energy module.
    """
    wf.paths = _resolve_simfolder(
        engine=engine,
        working_directory=working_directory,
        subdir=subdir,
    )
    wf.q = _commensurate_q_points_node(structure=structure, q_mesh=q_mesh)
    wf.n_atoms = _n_atoms_node(structure=structure)

    wf.md_renorm = calculate_phonon_md_renormalisation(
        structure=structure,
        engine=wf.paths.outputs.sub_engine,
        fc2_supercell_matrix=fc2_supercell_matrix,
        temperature=temperature,
        equilibration_steps=equilibration_steps,
        production_steps=production_steps,
        time_step=time_step,
        thermostat_time_constant=thermostat_time_constant,
        seed=seed,
        q_points=wf.q.outputs.q_points,
        phono3py_output=phono3py_output,
        power_spectra=False,
        keep_handles=True,  # we need .phonopy / .quasiparticle handles to extract data
    )
    wf.guarded = _guard_unphysical_frequencies(
        renormalised_frequencies=(
            wf.md_renorm.outputs.md_phonon_output.renormalised_frequencies
        ),
        harmonic_frequencies=(
            wf.md_renorm.outputs.md_phonon_output.harmonic_frequencies
        ),
    )
    wf.spectrum = _free_energy_from_spectrum(
        frequencies=wf.guarded.outputs.guarded_frequencies,
        q_weights=wf.q.outputs.q_weights,
        temperature=temperature,
        # Per-unit-cell-atom: primitive equals unitcell for fcc cubic; see docstring.
        n_atoms_primitive=wf.n_atoms.outputs.n_atoms,
    )
    wf.synthesis = _pack_anharmonic_dynaphopy_output(
        structure=structure,
        md_phonon_output=wf.md_renorm.outputs.md_phonon_output,
        q_weights=wf.q.outputs.q_weights,
        free_energy_per_atom=wf.spectrum.outputs.free_energy_per_atom,
        entropy_per_atom=wf.spectrum.outputs.entropy_per_atom,
        cv_per_atom=wf.spectrum.outputs.cv_per_atom,
        temperature=temperature,
        q_mesh=q_mesh,
        n_guarded=wf.guarded.outputs.n_guarded,
        simfolder=wf.paths.outputs.simfolder,
        keep_handles=keep_handles,
    )
    return wf.synthesis.outputs.free_energy_output


@pwf.as_function_node("per_T_outputs")
def _sweep_dynaphopy_over_T(
    structure: Atoms,
    engine: Engine,
    fc2_supercell_matrix,
    temperatures,
    equilibration_steps: int,
    production_steps: int,
    time_step: float,
    thermostat_time_constant: float,
    seed,
    q_mesh,
    working_directory: str,
) -> list:
    """Run ``anharmonic_free_energy_dynaphopy`` at each T; return list of outputs.

    The loop over ``temperatures`` lives in a function-node (rather than the
    macro body) so it executes with concrete values at run time instead of
    against pyiron_workflow ``UserInput`` proxies at graph-build time.
    """
    outputs: list = []
    for i, T in enumerate(temperatures):
        sub_seed = None if seed is None else int(seed) + i
        sub_wf = anharmonic_free_energy_dynaphopy(
            structure=structure,
            engine=engine,
            fc2_supercell_matrix=fc2_supercell_matrix,
            temperature=float(T),
            equilibration_steps=equilibration_steps,
            production_steps=production_steps,
            time_step=time_step,
            thermostat_time_constant=thermostat_time_constant,
            seed=sub_seed,
            q_mesh=q_mesh,
            working_directory=working_directory,
            subdir=f"T_{i:03d}_{float(T):.1f}K",
            keep_handles=False,
        )
        result = sub_wf.run()
        # Unwrap DotDict if pyiron_workflow returned the macro output that way.
        if isinstance(result, dict):
            result = result["free_energy_output"]
        outputs.append(result)
    return outputs


@pwf.as_function_node("free_energy_output")
def _stack_tdi_outputs(
    per_T_outputs: list,
    structure,
    temperatures,
) -> FreeEnergyOutput:
    """Aggregate independent-T dynaphopy free energies into one FreeEnergyOutput.

    Computes S(T) = -∂F/∂T and Cv(T) = -T ∂²F/∂T² via ``np.gradient`` central
    differences over the supplied ``temperatures``. With <2 temperatures S is
    NaN; with <3, Cv is NaN and ``report["derivative_warning"]`` is True.
    """
    T = np.asarray(temperatures, dtype=float)
    F = np.asarray([o.free_energy for o in per_T_outputs], dtype=float)
    n_T = T.size

    dF_dT = None
    if n_T >= 2:
        dF_dT = np.gradient(F, T)
        S = -dF_dT
    else:
        S = np.full(n_T, np.nan)
    if n_T >= 3 and dF_dT is not None:
        d2F_dT2 = np.gradient(dF_dT, T)
        Cv = -T * d2F_dT2
    else:
        Cv = np.full(n_T, np.nan)

    elements = (
        list(dict.fromkeys(structure.get_chemical_symbols()))
        if hasattr(structure, "get_chemical_symbols")
        else ["?"]
    )
    renorm = np.stack(
        [np.asarray(o.renormalised_frequencies) for o in per_T_outputs], axis=0
    )
    lw = np.stack([np.asarray(o.linewidths) for o in per_T_outputs], axis=0)

    derivative_warning = n_T < 3
    return FreeEnergyOutput(
        mode="anharmonic_dynaphopy_tdi",
        reference_phase="solid",
        free_energy=float(F[0]),
        free_energy_error=0.0,
        temperature=float(T[0]),
        pressure=0.0,
        n_atoms=len(structure),
        elements=elements,
        simfolder=per_T_outputs[0].simfolder if per_T_outputs else "",
        report={
            "method": "anharmonic_dynaphopy_tdi",
            "derivative_warning": bool(derivative_warning),
            "per_T_md_health": [o.report.get("md_health") for o in per_T_outputs],
        },
        temperature_array=T,
        free_energy_array=F,
        entropy_array=S,
        heat_capacity_array=Cv,
        renormalised_frequencies_per_T=renorm,
        linewidths_per_T=lw,
    )


@pwf.api.as_macro_node("free_energy_output")
def anharmonic_free_energy_dynaphopy_tdi(
    wf,
    structure: Atoms,
    engine: Engine,
    fc2_supercell_matrix,
    temperatures=(100.0, 200.0, 300.0, 400.0, 500.0),
    equilibration_steps: int = 2000,
    production_steps: int = 10000,
    time_step: float = 1.0,
    thermostat_time_constant: float = 100.0,
    seed: int | None = None,
    q_mesh=(11, 11, 11),
    working_directory: str = ".",
    subdir: str = "anharmonic_free_energy_dynaphopy_tdi",
    keep_handles: bool = False,
):
    """Anharmonic F_anharm(T) on a T grid — renormalised-harmonic at each T.

    Runs ``anharmonic_free_energy_dynaphopy`` independently at each requested
    temperature (each T gets its own MD trajectory + dynaphopy projection +
    harmonic-formula sum on the renormalised spectrum), then stacks the
    per-T results into a single ``FreeEnergyOutput`` and derives S(T) and
    Cv(T) by finite-differencing F(T) with ``np.gradient``.

    Despite the ``_tdi`` suffix, this is *not* a full ⟨∂H/∂λ⟩ thermodynamic
    integration over a coupling parameter — see spec
    ``docs/design/specs/2026-05-15-free-energy-consolidation-design.md`` for
    the renormalised-harmonic-over-T justification.
    """
    wf.paths = _resolve_simfolder(
        engine=engine,
        working_directory=working_directory,
        subdir=subdir,
    )
    wf.sweep = _sweep_dynaphopy_over_T(
        structure=structure,
        engine=engine,
        fc2_supercell_matrix=fc2_supercell_matrix,
        temperatures=temperatures,
        equilibration_steps=equilibration_steps,
        production_steps=production_steps,
        time_step=time_step,
        thermostat_time_constant=thermostat_time_constant,
        seed=seed,
        q_mesh=q_mesh,
        working_directory=wf.paths.outputs.simfolder,
    )
    wf.stack = _stack_tdi_outputs(
        per_T_outputs=wf.sweep.outputs.per_T_outputs,
        structure=structure,
        temperatures=temperatures,
    )
    return wf.stack.outputs.free_energy_output
