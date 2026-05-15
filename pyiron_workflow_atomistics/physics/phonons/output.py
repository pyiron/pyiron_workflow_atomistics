"""PhononOutput dataclass — the structured result of a phonon workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from ase import Atoms


@dataclass
class PhononOutput:
    """Structured result of a phonon thermal-conductivity calculation.

    Required fields are always populated. Optional fields are populated only
    when the corresponding macro flag is on:
        mode_resolved=True       → q_points, frequencies, group_velocities,
                                   mode_kappa, gamma, gruneisen
        harmonic_observables=True → band_structure, dos, free_energy
        keep_handles=True        → fc2, fc3, phono3py
    """

    structure: Atoms
    fc2_supercell_matrix: np.ndarray  # (3, 3) int
    fc3_supercell_matrix: np.ndarray  # (3, 3) int
    temperatures: np.ndarray  # (n_T,) K
    kappa: np.ndarray  # (n_T, 3, 3) W/m·K
    converged: bool

    q_points: np.ndarray | None = None  # (n_q, 3) reduced
    frequencies: np.ndarray | None = None  # (n_q, n_band) THz
    group_velocities: np.ndarray | None = None  # (n_q, n_band, 3)
    mode_kappa: np.ndarray | None = None  # (n_T, n_q, n_band, 6) Voigt
    gamma: np.ndarray | None = None  # (n_T, n_q, n_band) linewidths
    gruneisen: np.ndarray | None = None  # (n_q, n_band)

    band_structure: dict | None = None
    dos: dict | None = None
    free_energy: dict | None = None

    fc2: np.ndarray | None = None
    fc3: np.ndarray | None = None
    phono3py: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict of every field (ASE/phono3py objects by reference)."""
        return asdict(self)


@dataclass
class MdPhononOutput:
    """Structured result of a dynaphopy MD-trajectory mode-projection workflow.

    Required fields are always populated. Optional fields are populated only
    when the corresponding macro flag is on:
        power_spectra=True  → power_spectra, frequency_grid
        keep_handles=True   → quasiparticle, dynamics, phonopy

    MD health diagnostics
    ---------------------
    The two fields below let you sanity-check the NVT segment that drove the
    projection. Anharmonic renormalisation results are only as good as the
    underlying trajectory; if the diagnostics look bad, treat the
    ``renormalised_frequencies`` and ``linewidths`` as suspect.

    md_temperature_mean : float
        Time-averaged kinetic temperature over the production segment, in K.
        Healthy: within ~3% of the requested ``temperature``. Drift larger
        than that means the thermostat coupling time is too long, the
        production segment is too short to equilibrate, or the chosen
        integrator is leaking energy. Rerun with adjusted
        ``thermostat_time_constant`` or longer ``equilibration_steps``.

    md_temperature_std : float
        Std-dev of the instantaneous kinetic temperature over the production
        segment, in K. For a Langevin NVT, the expected fluctuation scales
        as ``T * sqrt(2 / (3 * N))`` where N is atom count — e.g. for
        N=32 atoms at T=300 K, σ_T ≈ 43 K. Values dramatically larger or
        smaller than this rule of thumb indicate sampling or coupling
        problems.

    Call ``out.check_md_health()`` to get a structured pass/fail summary.
    """

    structure: Atoms
    fc2_supercell_matrix: np.ndarray  # (3, 3) int
    temperature: float  # K (target of the NVT run)
    q_points: np.ndarray  # (n_q, 3) reduced — actually used
    harmonic_frequencies: np.ndarray  # (n_q, n_band) THz — pre-renormalisation
    renormalised_frequencies: np.ndarray  # (n_q, n_band) THz — fitted
    linewidths: np.ndarray  # (n_q, n_band) THz FWHM
    converged: bool  # all Lorentzian fits converged

    n_md_steps: int  # production-only count
    time_step_fs: float
    md_temperature_mean: float
    md_temperature_std: float

    power_spectra: np.ndarray | None = None  # (n_q, n_band, n_freq_bins)
    frequency_grid: np.ndarray | None = None  # (n_freq_bins,) THz

    quasiparticle: Any | None = None  # dynaphopy.Quasiparticle
    dynamics: Any | None = None  # dynaphopy.Dynamics
    phonopy: Any | None = None  # phonopy.Phonopy view used

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict of every field (heavy objects by reference)."""
        return asdict(self)

    def check_md_health(self, drift_tolerance: float = 0.03) -> tuple[bool, list[str]]:
        """Sanity-check the MD segment that drove the projection.

        Parameters
        ----------
        drift_tolerance
            Allowed relative drift between ``md_temperature_mean`` and the
            requested ``temperature``. Default 3%.

        Returns
        -------
        (is_healthy, issues)
            ``is_healthy`` is True iff no warnings fired. ``issues`` is a list
            of human-readable strings naming each issue.
        """
        issues: list[str] = []

        drift = abs(self.md_temperature_mean - self.temperature) / self.temperature
        if drift > drift_tolerance:
            issues.append(
                f"⟨T⟩ drift {drift:.1%} exceeds tolerance {drift_tolerance:.0%}: "
                f"requested {self.temperature:.1f} K, measured "
                f"{self.md_temperature_mean:.1f} K"
            )

        n_supercell_atoms = len(self.structure) * int(
            round(abs(np.linalg.det(self.fc2_supercell_matrix)))
        )
        expected_std = self.temperature * np.sqrt(2.0 / (3.0 * n_supercell_atoms))
        if expected_std > 0:
            ratio = self.md_temperature_std / expected_std
            if ratio < 0.5 or ratio > 2.0:
                issues.append(
                    f"σ_T = {self.md_temperature_std:.1f} K is {ratio:.2f}× the "
                    f"Langevin NVT expectation ({expected_std:.1f} K for "
                    f"{n_supercell_atoms} atoms at {self.temperature:.1f} K); "
                    f"thermostat coupling may be wrong"
                )
        return (not issues, issues)
