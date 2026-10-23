"""Internal helpers connecting pyiron_workflow_atomistics to calphy.

Public callers should not import from this module — use the function
nodes in ``physics/free_energy/calphy.py``. Everything here is
considered private API.
"""

from __future__ import annotations

import glob
import os
import shlex
from dataclasses import MISSING, fields
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import write as ase_write

if TYPE_CHECKING:
    from pyiron_workflow_lammps.engine import LammpsEngine

_LAUNCHERS = ("mpirun", "mpiexec", "srun")
_LAUNCHER_CORE_FLAGS = ("-np", "-n")
# LAMMPS switches calphy sets on every launch itself (``-in <segment>
# -log <segment log> -screen none``), long and short forms. A value given in
# the engine command is dropped.
_BINARY_TAIL_FLAGS = ("-in", "-i", "-log", "-l", "-screen", "-sc")
# LAMMPS switches that change what a run does in ways calphy cannot drive:
# multi-partition runs, conversion-only modes, help, and skipping ``run``.
_REJECTED_TAIL_FLAGS = (
    "-partition",
    "-p",
    "-plog",
    "-pscreen",
    "-mpicolor",
    "-m",
    "-restart2data",
    "-r2data",
    "-restart2dump",
    "-r2dump",
    "-skiprun",
    "-sr",
    "-help",
    "-h",
)


def _split_lammps_command(cmd: str) -> tuple[str, str | None, int, str]:
    """Parse a LAMMPS launcher command into (binary, mpi_executable, cores, cmdargs).

    Accepts:
      "lmp"                                   -> ("lmp", None, 1, "")
      "lmp -in in.lmp -log log.lammps"        -> ("lmp", None, 1, "")
      "mpirun -np 4 lmp"                      -> ("lmp", "mpirun", 4, "")
      "mpirun --bind-to none -np 2 lmp"       -> ("lmp", "mpirun --bind-to none", 2, "")
      "lmp -k on g 1 -sf kk -pk kokkos"       -> ("lmp", None, 1, "-k on g 1 -sf kk -pk kokkos")

    Tokens after the binary become ``cmdargs``, which calphy (``md.cmdargs``)
    appends to every ``lmp`` launch, e.g. accelerator switches. ``-in``,
    ``-log`` and ``-screen`` (and their values) are dropped, because calphy
    sets them on every launch itself. Switches calphy cannot drive
    (``_REJECTED_TAIL_FLAGS``) and tokens containing whitespace (calphy splits
    ``cmdargs`` on whitespace) are rejected.
    """
    if not cmd or not cmd.strip():
        raise ValueError("Empty LAMMPS command")

    tokens = shlex.split(cmd)
    i = 0

    mpi_parts: list[str] = []
    cores = 1
    has_launcher = tokens[0] in _LAUNCHERS

    if has_launcher:
        mpi_parts.append(tokens[i])
        i += 1
        while i < len(tokens):
            tok = tokens[i]
            if tok in _LAUNCHER_CORE_FLAGS:
                if i + 1 >= len(tokens):
                    raise ValueError(f"Missing core count after {tok!r}")
                try:
                    cores = int(tokens[i + 1])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid core count after {tok!r}: {tokens[i + 1]!r}"
                    ) from exc
                i += 2
                continue
            if tok.startswith("-"):
                # generic launcher flag with or without value
                mpi_parts.append(tok)
                # Heuristic: consume a value if the next token doesn't start with '-'.
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    # Only consume if it's NOT the LAMMPS binary. Binaries
                    # are named "lmp", "lmp_*", or end with "/lmp" or
                    # similar; flag values are arbitrary strings. The
                    # convention this parser supports is that launcher
                    # flag values come before the binary, so we treat
                    # the FIRST non-dash, non-binary-looking token as the
                    # value and break to look for the binary.
                    candidate = tokens[i + 1]
                    if _looks_like_lammps_binary(candidate):
                        i += 1
                        break
                    mpi_parts.append(candidate)
                    i += 2
                    continue
                i += 1
                continue
            # First non-launcher, non-flag token is the binary
            break

    if i >= len(tokens):
        raise ValueError(f"No LAMMPS binary in command: {cmd!r}")

    binary = tokens[i]
    if not _looks_like_lammps_binary(binary):
        raise ValueError(
            f"Token {binary!r} does not look like a LAMMPS binary "
            f"(expected name containing 'lmp' or 'lammps')"
        )
    i += 1

    # Tail: drop the switches calphy sets itself, refuse the ones it cannot
    # drive, and forward the rest as calphy's md.cmdargs.
    cmdargs: list[str] = []
    rejected: list[str] = []
    while i < len(tokens):
        tok = tokens[i]
        if tok in _BINARY_TAIL_FLAGS:
            if i + 1 >= len(tokens):
                rejected.append(tok)
                break
            i += 2
            continue
        if tok in _REJECTED_TAIL_FLAGS or any(c.isspace() for c in tok):
            rejected.append(tok)
        else:
            cmdargs.append(tok)
        i += 1
    if rejected:
        raise ValueError(
            f"Unsupported LAMMPS arguments in LammpsEngine.command {cmd!r}: "
            f"{rejected}. calphy launches LAMMPS once per run segment with "
            f"its own -in/-log/-screen and appends the remaining arguments "
            f"verbatim (split on whitespace), so multi-partition, conversion "
            f"and help switches, a -in/-log/-screen without a value, and "
            f"arguments containing whitespace cannot be used."
        )

    mpi = " ".join(mpi_parts) if mpi_parts else None
    return binary, mpi, cores, " ".join(cmdargs)


def _looks_like_lammps_binary(token: str) -> bool:
    """Heuristic: a LAMMPS binary's last path component contains 'lmp' or 'lammps'.

    Limitation: this is a substring match, so an MPI launcher flag whose
    *value* happens to contain "lmp" (e.g. ``mpirun --nodefile lmp_file -np
    4 lmp``) will trip the heuristic and the parser will misidentify the
    flag value as the binary. The spec accepts this trade-off because (a)
    flag values containing the substring "lmp" are extremely rare in
    practice, and (b) tightening the rule (e.g. requiring exact "lmp",
    "lammps", or "lmp_*") would break legitimate user setups with binaries
    like ``mylammps`` or ``mpilmp``. Users hitting this case can quote the
    binary path with a different name, or omit the offending launcher flag.
    """
    name = token.rsplit("/", 1)[-1].lower()
    return "lmp" in name or "lammps" in name


def _check_calphy_launcher(cmd: str, mpi: str | None) -> None:
    """Refuse launcher forms calphy>=2 cannot reproduce.

    calphy>=2 starts every LAMMPS segment itself as ``[mpi_executable,
    "-np", str(cores), lmp, ...]`` (``calphy.runner.ExecutableRunner``),
    resolves ``mpi_executable`` with ``shutil.which`` and only uses it when
    ``queue.cores > 1``. So launcher flags cannot be forwarded, ``srun``
    (which has no ``-np``) cannot be driven, and a launcher without an
    explicit core count would silently run serially. Refuse each case
    rather than run something other than what the engine command says.
    """
    if mpi is None:
        return
    launcher, *flags = mpi.split()
    if flags:
        raise ValueError(
            f"calphy>=2 accepts only a bare MPI launcher (e.g. 'mpirun -np 4 "
            f"lmp'); the launcher flags {flags} in {cmd!r} cannot be "
            f"forwarded. Remove them or set them via the launcher's "
            f"environment (e.g. OMPI_MCA_*)."
        )
    if launcher == "srun":
        raise ValueError(
            f"calphy>=2 launches parallel LAMMPS as '<launcher> -np <cores> "
            f"lmp', which srun does not accept; use 'mpirun -np N lmp' or "
            f"'mpiexec -np N lmp' instead of {cmd!r}."
        )
    if not any(tok in _LAUNCHER_CORE_FLAGS for tok in shlex.split(cmd)):
        raise ValueError(
            f"{cmd!r} names an MPI launcher but no core count; calphy>=2 "
            f"needs it explicitly (e.g. '{launcher} -np 4 lmp') and would "
            f"otherwise run LAMMPS serially."
        )


_ENGINE_CARVE_OUTS = frozenset(
    {
        "EngineInput",  # required to construct; ignored by calphy
        "mode",  # init=False, derived from EngineInput
        "working_directory",  # adapter sets its own simfolder
        "command",  # the one field we actually use
        "calc_fn",  # internal engine state, mutated by __post_init__
        "calc_fn_kwargs",  # internal engine state, mutated by __post_init__ (None -> {})
        "parse_fn",  # internal engine state, mutated by __post_init__
        "parse_fn_kwargs",  # internal engine state, mutated by __post_init__ (None -> {})
    }
)


def _validate_engine_only_command(engine: LammpsEngine) -> None:
    """Refuse LammpsEngine fields that the calphy adapter cannot honor.

    Only ``engine.command`` is consumed. Every other field that has been
    changed from its dataclass default is rejected, because calphy
    generates its own LAMMPS input from the supplied LammpsPotential
    and silently ignores everything else on the engine.
    """
    overridden: list[str] = []
    for f in fields(engine):
        if f.name in _ENGINE_CARVE_OUTS:
            continue
        default = _resolve_default(f)
        current = getattr(engine, f.name)
        if not _equals_default(current, default):
            overridden.append(f.name)

    if overridden:
        raise ValueError(
            f"calphy adapter only reads LammpsEngine.command. The "
            f"following non-default fields were set: {sorted(overridden)}. "
            f"calphy generates its own LAMMPS input; setting these "
            f"silently has no effect on the free-energy result. Pass "
            f"the potential via `potential=LammpsPotential(...)` and "
            f"construct a minimal engine:\n"
            f"  LammpsEngine(\n"
            f"      EngineInput=CalcInputStatic(),\n"
            f"      command='mpirun -np 4 lmp',\n"
            f"  )"
        )


def _resolve_default(f):
    """Return a dataclass field's default value, evaluating default_factory."""
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:  # type: ignore[misc]
        return f.default_factory()  # type: ignore[misc]
    return MISSING


def _equals_default(current, default) -> bool:
    """Tolerant equality: MISSING means no default -> cannot judge -> accept."""
    if default is MISSING:
        return True
    return current == default


def _validate_structure(structure: Atoms) -> None:
    """Refuse structures calphy cannot consume meaningfully.

    calphy assumes a fully periodic 3D supercell with positive volume
    and at least one atom. Anything else either crashes calphy mid-run
    or silently produces meaningless free energies (open boundaries
    don't have a well-defined Frenkel-Ladd reference).
    """
    if len(structure) == 0:
        raise ValueError("structure is empty (zero atoms)")
    pbc = tuple(bool(p) for p in structure.pbc)
    if pbc != (True, True, True):
        raise ValueError(
            f"calphy free-energy workflows require fully periodic 3D "
            f"PBC; got pbc={pbc}"
        )
    try:
        volume = structure.get_volume()
    except ValueError as exc:
        raise ValueError(
            "structure has non-positive volume; "
            "calphy will refuse to integrate against it"
        ) from exc
    if volume <= 0.0:
        raise ValueError(
            f"structure has non-positive volume ({volume}); "
            f"calphy will refuse to integrate against it"
        )


# ---------------------------------------------------------------------------
# _build_calphy_calculation — per-mode kwarg fan-out + data-file writer
# ---------------------------------------------------------------------------


def _atoms_element_order(structure: Atoms) -> list[str]:
    """Same first-occurrence ordering rule LammpsEngine uses."""
    return list(dict.fromkeys(structure.get_chemical_symbols()))


def _atoms_to_lammps_data(
    structure: Atoms, path: str, element_order: list[str]
) -> None:
    """Write a LAMMPS data file with type ordering matching element_order.

    ase >=3.23 honors ``specorder`` for lammps-data writes; older versions
    silently use alphabetical order. We require ase==3.28.0 (see
    pyproject.toml) so specorder is guaranteed available.
    """
    ase_write(
        path,
        structure,
        format="lammps-data",
        specorder=element_order,
        atom_style="atomic",
    )


def _build_calphy_calculation(
    *,
    mode: Literal[
        "fe",
        "ts",
        "tscale",
        "pscale",
        "melting_temperature",
        "alchemy",
        "composition_scaling",
    ],
    structure: Atoms,
    potential,  # LammpsPotential
    lammps_engine,  # LammpsEngine
    working_directory: str,
    # mode-generic kwargs (each public node forwards the subset it needs)
    temperature: float | None = None,
    temperature_range: tuple[float, float] | None = None,
    pressure: float = 0.0,
    pressure_range: tuple[float, float] | None = None,
    reference_phase: Literal["solid", "liquid"] | None = None,
    temperature_guess: float | None = None,
    melting_step: int = 200,
    melting_max_attempts: int = 5,
    pair_style_target: str | None = None,
    pair_coeff_target: str | None = None,
    output_chemical_composition: dict[str, int] | None = None,
    n_equilibration_steps: int = 25_000,
    n_switching_steps: int = 50_000,
    n_iterations: int = 1,
    npt: bool = True,
    equilibration_control: Literal["nose-hoover", "berendsen"] = "nose-hoover",
):
    """Build a ``calphy.input.Calculation`` from the user kwargs.

    Returns the validated Pydantic model, ready to feed to
    :func:`_run_calphy_job`. Writes the LAMMPS data file to
    ``{working_directory}/lammps.data`` as a side effect; calphy's validator
    then copies it to ``<identifier>.0.data`` in the *current* directory,
    which is why the public nodes chdir into the working directory first.
    """
    calphy = __import_calphy()

    element_order = _atoms_element_order(structure)
    mass = [float(atomic_masses[atomic_numbers[s]]) for s in element_order]

    data_path = os.path.join(working_directory, "lammps.data")
    os.makedirs(working_directory, exist_ok=True)
    _atoms_to_lammps_data(structure, data_path, element_order)

    binary, mpi, cores, cmdargs = _split_lammps_command(lammps_engine.command)
    _check_calphy_launcher(lammps_engine.command, mpi)

    kwargs: dict[str, Any] = {
        "mode": mode,
        "element": element_order,
        "mass": mass,
        "lattice": data_path,
        "file_format": "lammps-data",
        "pair_style": potential.pair_style,
        "pair_coeff": potential.pair_coeff,
        "potential_file": potential.potential_file,
        "n_equilibration_steps": n_equilibration_steps,
        "n_switching_steps": n_switching_steps,
        "n_iterations": n_iterations,
        "npt": npt,
        # calphy compares against the literal "nose-hoover"; any other
        # string silently selects the Berendsen branch.
        "equilibration_control": equilibration_control,
        "lammps_executable": binary,
        "mpi_executable": mpi,
        "queue": {
            "scheduler": "local",
            "cores": cores,
        },
        "md": {"cmdargs": cmdargs},
    }

    # Per-mode kwarg routing
    if mode == "fe":
        if temperature is None:
            raise ValueError("free_energy requires `temperature`")
        if reference_phase is None:
            raise ValueError("free_energy requires `reference_phase`")
        kwargs["temperature"] = float(temperature)
        kwargs["pressure"] = float(pressure)
        kwargs["reference_phase"] = reference_phase
    elif mode == "ts":
        if temperature_range is None or len(temperature_range) != 2:
            raise ValueError(
                "reversible_scaling_temperature requires "
                "`temperature_range=(lo, hi)`"
            )
        if reference_phase is None:
            raise ValueError("`reference_phase` is required")
        kwargs["temperature"] = [
            float(temperature_range[0]),
            float(temperature_range[1]),
        ]
        kwargs["pressure"] = float(pressure)
        kwargs["reference_phase"] = reference_phase
    elif mode == "pscale":
        if pressure_range is None or len(pressure_range) != 2:
            raise ValueError(
                "reversible_scaling_pressure requires " "`pressure_range=(lo, hi)`"
            )
        if temperature is None or reference_phase is None:
            raise ValueError("`temperature` and `reference_phase` are required")
        kwargs["temperature"] = float(temperature)
        kwargs["pressure"] = [float(pressure_range[0]), float(pressure_range[1])]
        kwargs["reference_phase"] = reference_phase
    elif mode == "melting_temperature":
        if temperature_guess is not None:
            if temperature_guess <= 0:
                raise ValueError("`temperature_guess` must be positive")
            kwargs["temperature"] = float(temperature_guess)
        kwargs["pressure"] = float(pressure)
        kwargs["melting_temperature"] = {
            "step": int(melting_step),
            "attempts": int(melting_max_attempts),
        }
    elif mode == "alchemy":
        if (
            temperature is None
            or pair_style_target is None
            or pair_coeff_target is None
        ):
            raise ValueError(
                "alchemy requires `temperature`, `pair_style_target`, "
                "and `pair_coeff_target`"
            )
        kwargs["temperature"] = float(temperature)
        kwargs["pressure"] = float(pressure)
        kwargs["reference_phase"] = "solid"
        kwargs["pair_style"] = [potential.pair_style, pair_style_target]
        kwargs["pair_coeff"] = [potential.pair_coeff, pair_coeff_target]
    elif mode == "composition_scaling":
        if temperature is None or output_chemical_composition is None:
            raise ValueError(
                "composition_scaling requires `temperature` and "
                "`output_chemical_composition`"
            )
        kwargs["temperature"] = float(temperature)
        kwargs["pressure"] = float(pressure)
        kwargs["reference_phase"] = "solid"
        input_composition = {
            el: structure.get_chemical_symbols().count(el) for el in element_order
        }
        output_composition = {
            el: int(n) for el, n in output_chemical_composition.items() if n
        }
        if output_composition == input_composition:
            raise ValueError(
                f"composition_scaling: output_chemical_composition "
                f"{dict(output_chemical_composition)} equals the composition of "
                f"`structure`; there is nothing to transform (use "
                f"`free_energy` for the free energy of this composition)."
            )
        kwargs["composition_scaling"] = {
            "output_chemical_composition": dict(output_chemical_composition),
        }
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    if mode == "melting_temperature":
        # calphy's MeltingTemp re-reads ``calc.inputfile`` / ``calc.kernel``
        # to spawn its solid and liquid ``ts`` sub-calculations, so a
        # programmatically built Calculation must be backed by a file.
        import yaml as _yaml

        input_path = os.path.join(working_directory, "calphy_input.yaml")
        with open(input_path, "w") as f:
            _yaml.safe_dump({"calculations": [kwargs]}, f)
        kwargs["inputfile"] = input_path
        kwargs["kernel"] = 0

    return calphy.input.Calculation(**kwargs)


def __import_calphy():
    """Import calphy lazily so the adapter module is importable without it."""
    from pyiron_workflow_atomistics.physics.free_energy._compat import (
        _require_calphy,
    )

    return _require_calphy()


def _setup_calculation(calc):
    """Build the calphy phase-job from a Calculation.

    Mirrors ``calphy.queuekernel.setup_calculation`` (calphy>=2.3):
    ``Solid`` / ``Liquid`` for ``fe``/``ts``/``tscale``/``pscale`` (by
    ``reference_phase``), ``Alchemy`` for ``alchemy`` *and*
    ``composition_scaling`` (the routine builds the
    ``CompositionTransformation`` itself), and ``MeltingTemp`` for
    ``melting_temperature``. The phase constructors write
    ``input_file.yaml`` into ``simfolder`` immediately, so it is
    materialised via ``calc.create_folders()`` first. ``MeltingTemp``
    needs no folder of its own: it logs to, and creates its solid and
    liquid sub-calculation folders in, the current directory, which is
    what its ``simfolder`` is set to.
    """
    calphy = __import_calphy()
    mode = getattr(calc, "mode", None)
    if mode == "melting_temperature":
        return calphy.MeltingTemp(calculation=calc, simfolder=os.getcwd())
    simfolder = calc.create_folders()
    ref = getattr(calc, "reference_phase", None)
    if mode in ("alchemy", "composition_scaling"):
        return calphy.Alchemy(calculation=calc, simfolder=simfolder)
    # fe / ts / tscale / pscale fork on reference_phase
    if ref == "liquid":
        return calphy.Liquid(calculation=calc, simfolder=simfolder)
    return calphy.Solid(calculation=calc, simfolder=simfolder)


def _run_calculation(job):
    """Run a calphy phase-job to completion via calphy's own routines.

    Mirrors ``calphy.queuekernel``: every mode maps 1:1 onto an upstream
    ``routine_*`` function (calphy>=2 runs averaging *and* its
    post-processing inside ``run_averaging`` and drives the ``lmp``
    binary itself), and ``melting_temperature`` calls
    ``MeltingTemp.calculate_tm``.
    """
    from calphy import MeltingTemp, routines

    if isinstance(job, MeltingTemp):
        try:
            job.calculate_tm()
        except ValueError as exc:
            if "Simulation folder" not in str(exc):
                raise
            # MeltingTemp names each solid/liquid sub-calculation after its
            # temperature bracket and refuses to reuse a folder.
            raise RuntimeError(
                f"calphy's melting-temperature search needed a sub-calculation "
                f"folder that already exists ({exc}). Either an earlier run "
                f"left it in the working directory, or the bracket search "
                f"oscillated: the liquid froze at the bottom of one bracket and "
                f"the solid melted at the top of the next, so it went back to "
                f"a bracket it had tried. Use a fresh working directory, and "
                f"for oscillation a smaller `step`, a closer "
                f"`temperature_guess`, a larger cell or longer sweeps."
            ) from exc
        return job

    mode = getattr(getattr(job, "calc", None), "mode", None)
    routine = {
        "fe": routines.routine_fe,
        "ts": routines.routine_ts,
        "tscale": routines.routine_tscale,
        "pscale": routines.routine_pscale,
        "alchemy": routines.routine_alchemy,
        "composition_scaling": routines.routine_composition_scaling,
    }.get(mode)
    if routine is None:
        raise ValueError(
            f"No calphy routine for mode={mode!r}. Supported: "
            "'fe', 'ts', 'tscale', 'pscale', 'alchemy', 'composition_scaling', "
            "'melting_temperature'."
        )
    return routine(job)


def _run_calphy_job(calc):
    """Dispatch a built ``Calculation`` through calphy.

    Returns ``(job, report)`` where ``report`` is the parsed
    ``report.yaml`` from ``job.simfolder``. Reading the file rather
    than scraping live attributes survives calphy minor-version
    changes that move fields around. ``melting_temperature`` writes no
    such file; its report is assembled by :func:`_melting_report`.
    """
    import yaml as _yaml

    job = _setup_calculation(calc)
    job = _run_calculation(job)
    if getattr(calc, "mode", None) == "melting_temperature":
        return job, _melting_report(job)
    report_path = os.path.join(job.simfolder, "report.yaml")
    try:
        with open(report_path) as f:
            report = _yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"calphy did not produce a report.yaml in {job.simfolder}; "
            f"the run may have failed silently. Inspect that directory "
            f"for partial artefacts."
        ) from exc
    except _yaml.YAMLError as exc:
        raise RuntimeError(
            f"Failed to parse {report_path}: {exc}. Inspect the file "
            f"for malformed YAML."
        ) from exc
    return job, report


def _melting_report(job) -> dict:
    """Summarise a finished ``MeltingTemp`` job as a report dict.

    calphy writes no top-level ``report.yaml`` for this mode; the result
    lives on the job (``calc_tm`` / ``tmerr``) and the solid and liquid
    ``ts`` sub-calculations keep their own reports. Their sweep-dissipation
    verdicts are carried along because calphy only *logs* a warning when a
    sweep was irreversible (a hidden phase change), which makes Tm
    unreliable.
    """
    tm = getattr(job, "calc_tm", None)
    err = getattr(job, "tmerr", None)
    phases = {}
    for name, sub in (("solid", "soljob"), ("liquid", "lqdjob")):
        sub_job = getattr(job, sub, None)
        if sub_job is None:
            continue
        phases[name] = {
            "simfolder": getattr(sub_job, "simfolder", None),
            "ts_dissipation": (
                float(sub_job.ediss) if hasattr(sub_job, "ediss") else None
            ),
            "ts_dissipation_high": bool(getattr(sub_job, "ediss_high", False)),
        }
    return {
        "results": {
            "melting_temperature": None if tm is None else float(tm),
            "melting_temperature_error": None if err is None else float(err),
            "ts_dissipation_high": any(
                p["ts_dissipation_high"] for p in phases.values()
            ),
        },
        "phases": phases,
    }


def _pack_free_energy_output(
    *,
    mode: str,
    job,
    report: dict,
    simfolder: str,
    structure: Atoms,
    reference_phase: str,
    temperature: float,
    pressure: float,
):
    """Build a :class:`FreeEnergyOutput` from a finished calphy job."""
    from pyiron_workflow_atomistics.physics.free_energy.outputs import (
        FreeEnergyOutput,
    )

    results = (report or {}).get("results", {})

    out = FreeEnergyOutput(
        mode=mode,
        reference_phase=reference_phase,
        free_energy=float(results.get("free_energy", float("nan"))),
        free_energy_error=float(results.get("error", float("nan"))),
        temperature=float(temperature),
        pressure=float(pressure),
        n_atoms=len(structure),
        elements=_atoms_element_order(structure),
        simfolder=os.path.abspath(simfolder),
        report=report or {},
    )

    if mode == "fe":
        einstein = results.get("einstein_crystal")
        if einstein is not None:
            out.einstein_free_energy = float(einstein)
    elif mode in ("ts", "tscale"):
        try:
            t_arr, f_arr = _load_rs_curve(simfolder)
            out.temperature_array = t_arr
            out.free_energy_array = f_arr
        except FileNotFoundError:
            # calphy already wrote `report.yaml` so the run succeeded;
            # the curve file is simply absent. Leave the arrays None.
            pass
    elif mode == "pscale":
        try:
            p_arr, f_arr = _load_rs_curve(simfolder, axis="pressure")
            out.pressure_array = p_arr
            out.free_energy_array = f_arr
        except FileNotFoundError:
            pass
    elif mode == "melting_temperature":
        # calphy stores the result as ``calc_tm`` / ``tmerr``
        tm = getattr(job, "calc_tm", None)
        err = getattr(job, "tmerr", None)
        out.melting_temperature = float(tm) if tm is not None else None
        out.melting_temperature_error = float(err) if err is not None else None
    elif mode == "composition_scaling":
        # calphy>=2's report.yaml no longer echoes the composition_scaling
        # input block, so read the target from the job's Calculation.
        raw = (report or {}).get("input", {}).get("composition_scaling")
        if not isinstance(raw, dict):
            raw = _composition_scaling_block(job)
        comp = raw.get("output_chemical_composition") if raw else None
        if isinstance(comp, dict):
            out.composition_path = [dict(comp)]
        elif isinstance(comp, list) and all(isinstance(c, dict) for c in comp):
            out.composition_path = list(comp)
        else:
            out.composition_path = None

    return out


def _composition_scaling_block(job) -> dict | None:
    """Return ``job.calc.composition_scaling`` as a plain dict, if present."""
    block = getattr(getattr(job, "calc", None), "composition_scaling", None)
    if block is None:
        return None
    if isinstance(block, dict):
        return block
    comp = getattr(block, "output_chemical_composition", None)
    return {"output_chemical_composition": comp} if comp is not None else None


def _load_rs_curve(
    simfolder: str,
    axis: Literal["temperature", "pressure"] = "temperature",
) -> tuple[np.ndarray, np.ndarray]:
    """Read calphy's reversible-scaling sweep file.

    Calphy writes ``temperature_sweep.dat`` for ``ts``/``tscale`` and
    ``pressure_sweep.dat`` for ``pscale``. Both are two-column whitespace-
    separated: independent variable, then free energy.

    Returns ``(x_array, free_energy_array)`` as numpy arrays.

    Raises
    ------
    FileNotFoundError
        If the expected sweep file is not present in ``simfolder`` —
        either calphy didn't run, or its filename convention changed.
    """
    expected = f"{axis}_sweep.dat"
    path = os.path.join(simfolder, expected)
    if not os.path.exists(path):
        # Tolerate calphy's older convention `*free_energy*.dat`
        candidates = glob.glob(os.path.join(simfolder, "*sweep*.dat"))
        if not candidates:
            raise FileNotFoundError(
                f"No reversible-scaling sweep file in {simfolder} "
                f"(looked for {expected} and *sweep*.dat)"
            )
        path = candidates[0]
    data = np.loadtxt(path)
    if data.ndim == 1:  # single row
        data = data[None, :]
    return data[:, 0], data[:, 1]
