"""Internal helper for extracting attribute lists from EngineOutput sequences."""

from __future__ import annotations

from collections.abc import Sequence


def extract_outputs_from_EngineOutputs(
    engine_outputs: Sequence,
    keys: list[str],
    only_converged: bool = True,
) -> dict[str, list]:
    """Pluck ``keys`` from a sequence of EngineOutput-like objects into per-key lists.

    Parameters
    ----------
    engine_outputs
        Iterable of EngineOutput (or duck-typed) instances.
    keys
        Attribute names to extract from each output.
    only_converged
        If True (default), skip outputs whose ``convergence`` attribute is
        falsy or missing.
    """
    extracted: dict[str, list] = {key: [] for key in keys}
    for output in engine_outputs:
        if only_converged and not getattr(output, "convergence", False):
            continue
        for key in keys:
            extracted[key].append(getattr(output, key, None))
    return extracted
