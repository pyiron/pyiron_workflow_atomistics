"""Internal helpers connecting pyiron_workflow_atomistics to calphy.

Public callers should not import from this module — use the function
nodes in ``physics/free_energy/calphy.py``. Everything here is
considered private API.
"""

from __future__ import annotations

import shlex

_LAUNCHERS = ("mpirun", "mpiexec", "srun")
_LAUNCHER_CORE_FLAGS = ("-np", "-n")
_BINARY_TAIL_FLAGS = ("-in", "-log")


def _split_lammps_command(cmd: str) -> tuple[str, str | None, int]:
    """Parse a LAMMPS launcher command into (binary, mpi_executable, cores).

    Accepts:
      "lmp"                                          -> ("lmp", None, 1)
      "lmp -in in.lmp -log log.lammps"               -> ("lmp", None, 1)
      "mpirun -np 4 lmp"                             -> ("lmp", "mpirun", 4)
      "mpirun --bind-to none -np 2 lmp"              -> ("lmp", "mpirun --bind-to none", 2)

    Rejects any tokens other than recognised launcher flags, `-np`/`-n`,
    and the trailing `-in <file>` / `-log <file>` pair (which calphy
    overwrites with its own paths anyway).
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

    # Tail may contain only `-in <file>` and `-log <file>`; calphy will
    # replace these with its own paths. Anything else is rejected.
    unknown: list[str] = []
    while i < len(tokens):
        tok = tokens[i]
        if tok in _BINARY_TAIL_FLAGS:
            if i + 1 >= len(tokens):
                unknown.append(tok)
                break
            i += 2
            continue
        unknown.append(tok)
        i += 1
    if unknown:
        raise ValueError(
            f"Unrecognized tokens in LammpsEngine.command: {unknown}. "
            f"The calphy adapter only accepts launcher + binary + "
            f"optional -in/-log; everything else is dropped silently "
            f"by calphy and the adapter refuses rather than mislead."
        )

    return binary, " ".join(mpi_parts) if mpi_parts else None, cores


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


from dataclasses import MISSING, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyiron_workflow_lammps.engine import LammpsEngine


_ENGINE_CARVE_OUTS = frozenset({
    "EngineInput",        # required to construct; ignored by calphy
    "mode",               # init=False, derived from EngineInput
    "working_directory",  # adapter sets its own simfolder
    "command",            # the one field we actually use
    "calc_fn",            # internal engine state, mutated by __post_init__
    "calc_fn_kwargs",     # internal engine state, mutated by __post_init__ (None -> {})
    "parse_fn",           # internal engine state, mutated by __post_init__
    "parse_fn_kwargs",    # internal engine state, mutated by __post_init__ (None -> {})
})


def _validate_engine_only_command(engine: "LammpsEngine") -> None:
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
