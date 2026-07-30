"""Guarded access to the optional ``gb_code`` dependency.

``gb_code`` is distributed on PyPI as ``lz-GB-code`` and is an *optional*
dependency: it is not on conda-forge, so keeping it out of the mandatory
requirements is what allows this project to be packaged there. Every module in
this package imports the ``gb_code`` names from here rather than directly, so a
missing install produces an actionable message instead of a bare
``ModuleNotFoundError: No module named 'gb_code'``.
"""

_MISSING = (
    "The grain-boundary CSL search needs the optional `gb_code` package "
    "(distributed on PyPI as `lz-GB-code`), which is not installed.\n\n"
    "    pip install 'pyiron_workflow_atomistics[gb]'\n\n"
    "`gb_code` is PyPI-only — there is no conda-forge package for it — so the "
    "conda distribution of pyiron_workflow_atomistics does not pull it in and "
    "it must be pip-installed even in a conda environment. Installing "
    "`lz-GB-code==0.1.0` directly works too."
)

try:
    import gb_code.csl_generator as csl_generator
    from gb_code.csl_generator import get_theta_m_n_list
    from gb_code.gb_generator import GB_character
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(_MISSING) from exc

__all__ = ["GB_character", "csl_generator", "get_theta_m_n_list"]
