"""Bicrystal — GB structure construction and Voronoi interstitial search.

Direct port of GRIP ``core/bicrystal.py`` (commit 8ff6a43).

Departures from upstream:
    1. ``__init__`` takes a ``GCOConfig`` instead of a ``struct``/``algo``
       dict pair. The ``struct`` dict was stored but never read after
       construction; the ``algo`` dict's keys match ``GCOConfig`` field
       names 1:1 in the fields this class consumes.
    2. ``self.debug`` is gone; ``print(...)`` calls become ``logger.debug``
       (or ``logger.warning`` for the GB-region-too-thin etc. cases).
    3. ``write_gb`` always dispatches through ``ase.io.write``; the
       lammps-data fast path is dropped (ASE handles it natively).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import write as ase_write
from ase.lattice.bravais import Lattice
from scipy.spatial import Voronoi

from .config import GCOConfig
from .interstitial import Interstitial

logger = logging.getLogger(__name__)

_Z_THRESH = 1e-3


class Bicrystal:
    """A bicrystal: two slabs that can be translated, replicated, defected,
    perturbed, and joined into a single GB structure.

    See ``docs/design/specs/2026-05-15-grand-canonical-gb-design.md``
    for the per-iteration data-flow context.
    """

    def __init__(
        self,
        lower: Lattice,
        upper: Lattice,
        config: GCOConfig,
        dlat: float,
        make_copy: bool = True,
    ) -> None:
        self.lower0 = lower
        self.upper0 = upper
        self.config = config
        self.dlat = dlat

        self.rxyz: tuple[int, int, int] = (1, 1, 1)
        self.dxyz: list[float] = [0.0, 0.0, 0.0]
        self.dupper = config.perturb_u
        self.dlower = config.perturb_l
        self.npp_u: int | None = None
        self.gbplane_ids_u: np.ndarray | None = None
        self.gbplane_pos_u: np.ndarray | None = None
        self.bounds: np.ndarray | None = None
        self.gb: Atoms | None = None
        self.relaxed: bool = False
        self.n: float | None = None
        self.Egb: float | None = None
        self.inter_n: int = 0
        self.interstitials: list[Interstitial] = []

        if make_copy:
            self.copy_ul()

    # ------------------------------------------------------------------
    # Lifecycle / state
    # ------------------------------------------------------------------

    def copy_ul(self) -> None:
        """Reset the working slabs from the parent slabs."""
        self.lower = self.lower0.copy()
        self.upper = self.upper0.copy()

    def __repr__(self) -> str:
        if self.relaxed:
            suffix = "relaxed"
        elif self.gb:
            suffix = "joined"
        else:
            suffix = "unjoined"
        return (
            f"{self.__class__.__name__}({self.lower.symbols}, "
            f"{self.upper.symbols}) {suffix}"
        )

    def __eq__(self, other) -> bool:
        try:
            return self.upper == other.upper and self.lower == other.lower
        except Exception:
            return self.upper0 == other.upper0 and self.lower0 == other.lower0

    def __bool__(self) -> bool:
        return bool(self.gb)

    def __len__(self) -> int:
        return len(self.lower) + len(self.upper) + self.inter_n

    @property
    def natoms(self) -> int:
        return self.__len__()

    @property
    def z(self) -> float:
        return self.gb.cell.lengths()[2]

    # ------------------------------------------------------------------
    # Manipulation
    # ------------------------------------------------------------------

    def replicate(self, xreps: int, yreps: int, zreps: int = 1) -> None:
        reps = (xreps, yreps, zreps)
        self.lower *= reps
        self.upper *= reps
        self.rxyz = reps

    def shift_upper(self, xshift: float, yshift: float, zshift: float = 0.0) -> None:
        shifts = [xshift, yshift, zshift]
        self.upper.positions += shifts
        self.dxyz = shifts

    def get_bounds(self, config: GCOConfig) -> None:
        lowerb = self.lower.cell[2, 2] - config.gb_thick
        upperb = self.upper.cell[2, 2] - config.gb_thick
        self.bounds = np.array([lowerb, upperb, config.pad])

    def get_gbplane_atoms_u(self) -> int:
        sorted_pos = self.upper.positions[self.upper.positions[:, 2].argsort()]
        min_top = sorted_pos.round(6)[0, 2]
        mask = self.upper.positions[:, 2] < (min_top + _Z_THRESH)
        if self.dlat > 0:
            logger.debug("Calculating Nplane atoms within %.6f Å", self.dlat)
            mask = self.upper.positions[:, 2] < (min_top + self.dlat - _Z_THRESH)
        else:
            logger.debug("dlat is %s <= 0, taking planar slice.", self.dlat)
        self.npp_u = int(mask.sum())
        logger.debug("Atoms per top crystal plane = %d", self.npp_u)

        self.gbplane_ids_u = np.where(mask)[0]
        self.gbplane_pos_u = self.upper.positions[self.gbplane_ids_u]
        return self.npp_u

    def make_vacancies_u(self, index_list: Sequence[int]) -> None:
        for idx in sorted(index_list, reverse=True):
            self.upper.pop(idx)

    def defect_upper(self, config: GCOConfig, rng: np.random.Generator) -> None:
        assert self.npp_u is not None, (
            "Must calculate the number of atoms per plane first! "
            "Call get_gbplane_atoms_u()"
        )
        n_vac = int(
            rng.integers(
                np.floor(self.npp_u * (1 - config.frac_max)),
                np.ceil(self.npp_u * (1 - config.frac_min)),
                endpoint=True,
            )
        )
        to_delete = rng.choice(self.gbplane_ids_u, size=n_vac, replace=False)
        self.make_vacancies_u(to_delete)
        n_udef = self.upper.get_global_number_of_atoms()
        logger.debug("%d atoms in defective cell after %d vacancies.", n_udef, n_vac)

        self.n = np.mod(n_udef, self.npp_u)
        assert self.n == np.mod(self.npp_u - n_vac, self.npp_u), (
            "Atoms were not deleted properly! "
            f"{n_udef} atoms in defective cell due to {n_vac} vacancies; "
            f"with {self.npp_u} atoms/plane, n={self.n} != "
            f"{np.mod(self.npp_u - n_vac, self.npp_u)}."
        )
        self.n /= self.npp_u

    def perturb_atoms(self, rng: np.random.Generator) -> None:
        mask_upper = self.upper.positions[:, 2] < self.config.gb_thick / 2
        n_u = int(mask_upper.sum())
        self.upper.positions[mask_upper, :] += self.dupper * rng.random([n_u, 3])

        mask_lower = (
            self.lower.positions[:, 2]
            > self.lower.cell[2, 2] - self.config.gb_thick / 2
        )
        n_l = int(mask_lower.sum())
        self.lower.positions[mask_lower, :] += self.dlower * rng.random([n_l, 3])

    def join_gb(self, config: GCOConfig, gb_normal: int = 2) -> None:
        offset = self.lower.cell[gb_normal, gb_normal] + config.gb_gap
        upper_copy = self.upper.copy()
        upper_copy.positions[:, gb_normal] += offset
        upper_copy.extend(self.lower.copy())
        upper_copy.cell[gb_normal, gb_normal] += offset + config.vacuum
        self.gb = upper_copy

    def write_gb(self, filename: str, format: str = "extxyz") -> None:
        assert self.gb is not None, "GB hasn't been created yet! Call join_gb() first."
        ase_write(filename, self.gb, format=format)

    # ------------------------------------------------------------------
    # Voronoi interstitial search
    # ------------------------------------------------------------------

    def get_edge_midpts(self, pts: list, rv: list) -> np.ndarray:
        emps = []
        for edge in rv:
            emps.append((np.array(pts[edge[0]]) + np.array(pts[edge[1]])) / 2)
        return np.array(emps)

    def compute_voronoi(
        self,
        struct0: Atoms,
        bounds: tuple,
        edge_midpoints: bool,
        reps: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if reps is None:
            reps = np.array([3, 3, 1])
        a, b, c = struct0.cell.lengths()
        struct = struct0 * reps
        pts = struct.positions - (reps // 2) * np.array([a, b, c])

        vor = Voronoi(pts)
        logger.debug("Voronoi diagram with %s repetitions.", reps)

        vert = np.round(vor.vertices, 6)
        logger.debug("Found %d total vertices.", len(vert))

        if edge_midpoints:
            rv_pos = [rv for rv in vor.ridge_vertices if (np.array(rv) >= 0).all()]
            emps = self.get_edge_midpts(vert, rv_pos)
            vert = np.concatenate((vert, emps), axis=0)
            logger.debug("Adding %d edge midpoints.", len(emps))

        mask = (
            (vert[:, 0] > 0)
            & (vert[:, 0] < a)
            & (vert[:, 1] > 0)
            & (vert[:, 1] < b)
            & (vert[:, 2] > bounds[0])
            & (vert[:, 2] < bounds[1])
        )
        v = vert[mask]
        v = v[v[:, 2].argsort()]
        logger.debug("After masking, %d points returned.", len(v))
        return v, pts

    def check_exist(
        self,
        exist: list,
        curr: np.ndarray,
        n1: int,
        tol: float | None = None,
    ) -> bool:
        if not tol:
            tol = min(curr[:n1]) * 2e-2
        diff = np.array([np.linalg.norm(x[:n1] - curr[:n1]) for x in exist])
        if len(diff) == 0:
            exist.append(curr)
            return False
        if (diff > tol).all():
            exist.append(curr)
            return False
        return True

    def classify_sites(
        self,
        sites: np.ndarray,
        pos: np.ndarray,
        top_n: int = 10,
        abs_tol: float = 6e-1,
        rel_tol: float = 5e-3,
        reset_index: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        labels = []
        nn_list = []
        nnd_list = []
        other: list[np.ndarray] = []
        tc = oc = tstrainc = ostrainc = trc = trstrainc = otherc = 0

        for v in sites:
            dists = np.sort(np.linalg.norm(pos - v, axis=1))[:top_n]
            vdist = np.abs(dists - dists[0])
            nn1 = int((vdist < abs_tol).sum())
            nn2 = int((vdist < rel_tol * dists[0]).sum())
            nn1_dist = vdist[:nn1].sum()
            same_dist = nn1_dist < rel_tol * dists[0]

            nn_list.append(nn2)
            nnd_list.append(dists[:nn1].round(6))

            exist = self.check_exist(other, dists, nn1, tol=None)
            if nn2 == 3:
                if same_dist:
                    if not exist:
                        trc += 1
                    labels.append(f"triangular{trc}")
                else:
                    if not exist:
                        tstrainc += 1
                    # PORT-NOTE: upstream uses trstrainc here but only ever
                    # increments tstrainc, so this label always reads as
                    # "tri_strain0". Preserved verbatim; track for future fix.
                    labels.append(f"tri_strain{trstrainc}")
            elif nn2 == 4:
                if same_dist:
                    if not exist:
                        tc += 1
                    labels.append(f"tetrahedral{tc}")
                else:
                    if not exist:
                        tstrainc += 1
                    labels.append(f"tetra_strain{tstrainc}")
            elif nn2 == 6:
                if same_dist:
                    if not exist:
                        oc += 1
                    labels.append(f"octahedral{oc}")
                else:
                    if not exist:
                        ostrainc += 1
                    labels.append(f"octa_strain{ostrainc}")
            else:
                if not exist:
                    otherc += 1
                labels.append(f"other{otherc}")

        df = pd.DataFrame(
            {
                "x": sites[:, 0],
                "y": sites[:, 1],
                "z": sites[:, 2],
                "label": labels,
                "nn": nn_list,
                "nnd": nnd_list,
            }
        )
        df.sort_values(by=["z", "y", "x"], ignore_index=True, inplace=True)
        unique = df.drop_duplicates(subset=["label", "nn"], keep="first")
        if reset_index:
            unique.reset_index(drop=True, inplace=True)
        return df, unique

    def find_interstitials(
        self,
        zbounds: tuple[float, float] | None = None,
        edges: bool = False,
        unique_sites: bool = False,
    ) -> list[Interstitial]:
        assert self.gb is not None, "GB hasn't been created yet! Call join_gb() first."
        if zbounds is None:
            zbounds = (self.bounds[0], self.z - self.bounds[1])
        logger.debug("Searching for interstitials between %s.", zbounds)
        v, pts = self.compute_voronoi(self.gb, zbounds, edges)
        df, unique = self.classify_sites(v, pts)
        self.interstitials = Interstitial.from_df(unique if unique_sites else df)
        return self.interstitials

    def swap_gb_interstitials(
        self,
        zbounds: tuple[float, float],
        rng: np.random.Generator,
    ) -> int:
        gb_mask = (self.gb.positions[:, 2] >= zbounds[0]) & (
            self.gb.positions[:, 2] <= zbounds[1]
        )
        gb_ind = np.where(gb_mask)[0]
        rng.shuffle(gb_ind)

        if len(gb_ind) < self.config.inter_n:
            if len(gb_ind) <= len(self.interstitials):
                logger.warning(
                    "Only %d GB atoms to swap (requested %d).",
                    len(gb_ind),
                    self.config.inter_n,
                )
            else:
                logger.warning(
                    "Only %d interstitial sites to swap (requested %d).",
                    len(self.interstitials),
                    self.config.inter_n,
                )

        swapped_n = min(self.config.inter_n, len(self.interstitials), len(gb_ind))
        for i in range(swapped_n):
            self.gb[gb_ind[i]].position = self.interstitials[i].position()
        return swapped_n

    def find_and_swap_inters(self, rng: np.random.Generator) -> int:
        if self.config.inter_n > 0 and rng.random() < self.config.inter_p:
            zmid = self.lower0.cell[2, 2] + self.config.gb_gap / 2
            zbounds = (zmid - self.config.inter_t, zmid + self.config.inter_t)
            inters = self.find_interstitials(
                zbounds=zbounds, unique_sites=self.config.inter_u
            )
            if self.config.inter_r:
                rng.shuffle(inters)
            logger.debug("Found %d interstitial sites.", len(inters))
            if inters:
                logger.debug("First site: %s", inters[0])

            zbounds2 = (
                zmid - 2 * self.config.inter_t,
                zmid + 2 * self.config.inter_t,
            )
            return self.swap_gb_interstitials(zbounds=zbounds2, rng=rng)
        return 0
