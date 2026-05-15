"""dynaphopy MD-trajectory anharmonic phonon renormalisation workflow.

The single user-facing entry point is
:func:`calculate_phonon_md_renormalisation`.

Built on top of dynaphopy via a thin wrapper that exposes its functionality
as pyiron_workflow function-nodes and macros. The upstream package's name
is the authoritative source for behaviour and bug reports; this file
routes inputs/outputs through the pyiron_workflow Engine Protocol.
"""

from __future__ import annotations
