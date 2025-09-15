import pyiron_workflow_atomistics.gb.gb_code.csl_generator as csl
from pyiron_workflow_atomistics.gb.gb_code.csl_generator import get_theta_m_n_list
import numpy as np
import pandas as pd
from math import degrees
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pyiron_snippets.logger import logger


def _construct_gb_for_sigma(args):
    """
    Helper to construct GB data for a single sigma.
    """
    sigma, theta, m, n, axis, basis, lim_plane_index = args
    R = csl.rot(axis, theta)
    M1, M2 = csl.Create_minimal_cell_Method_1(sigma, axis, R)
    V1_list, V2_list, M_list, Gb_types = csl.Create_Possible_GB_Plane_List(
        axis, m, n, lim_plane_index
    )

    Number_atoms_list = [
        csl.Find_Orthogonal_cell(basis, axis, m, n, V1)[2] for V1 in V1_list
    ]

    V1_list = [tuple(V1) for V1 in V1_list]
    V2_list = [tuple(V2) for V2 in V2_list]

    return pd.DataFrame(
        {
            "Axis": [axis.tolist()] * len(Gb_types),
            "Sigma": [sigma] * len(Gb_types),
            "m": [m] * len(Gb_types),
            "n": [n] * len(Gb_types),
            "GB1": V1_list,
            "GB2": V2_list,
            "Theta (deg)": [degrees(theta)] * len(Gb_types),
            "Type": Gb_types,
            "n_atoms": Number_atoms_list,
        }
    )


def _construct_gbcode_df_for_axis(
    sigma_list: list[int],
    theta_list: list[float],
    m_list: list[np.ndarray],
    n_list: list[np.ndarray],
    axis: np.ndarray,
    basis: str = "fcc",
    lim_plane_index: int = 3,
    max_workers: int = None,
) -> pd.DataFrame:
    """
    Construct GB DataFrame for each sigma in parallel.

    Parameters
    ----------
    sigma_list : list of int
        List of Sigma values.
    theta_list : list of float
        Corresponding rotation angles (radians).
    m_list, n_list : list of np.ndarray
        Corresponding CSL m and n vectors.
    axis : np.ndarray
        Rotation axis.
    basis : str
        Crystal basis ('fcc', 'bcc', etc.).
    lim_plane_index : int
        Limit for GB plane enumeration.
    max_workers : int or None
        Number of parallel workers (defaults to cpu count or sigma count).

    Returns
    -------
    pd.DataFrame
        Combined DataFrame of all GB entries.
    """
    assert len(sigma_list) == len(theta_list) == len(m_list) == len(n_list)

    # Prepare arguments per sigma
    args = [
        (
            sigma_list[i],
            theta_list[i],
            m_list[i],
            n_list[i],
            axis,
            basis,
            lim_plane_index,
        )
        for i in range(len(sigma_list))
    ]

    # Determine number of workers
    if max_workers is None:
        max_workers = min(cpu_count(), len(args))

    # Parallel execution with progress bar
    with Pool(processes=max_workers) as pool:
        dfs = list(
            tqdm(
                pool.imap(_construct_gb_for_sigma, args),
                total=len(args),
                desc="Constructing GBs",
            )
        )

    # Concatenate all results
    return pd.concat(dfs, ignore_index=True)


def _deduplicate_gbcode_df_miller_indices_equivalent(df: pd.DataFrame) -> pd.DataFrame:
    def canonical_gb(row):
        import itertools

        # 1) generate the 48 orientation matrices for a cubic axes system
        orients = []
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product([1, -1], repeat=3):
                M = np.zeros((3, 3), int)
                for i, p in enumerate(perm):
                    M[i, p] = signs[i]
                orients.append(M)
        v1 = np.array(row["GB1"])
        v2 = np.array(row["GB2"])
        candidates = []
        # 2) apply every orientation
        for M in orients:
            Mv1, Mv2 = M.dot(v1), M.dot(v2)
            # 3) allow flipping each plane independently
            for s1, s2 in itertools.product([1, -1], repeat=2):
                sv1, sv2 = s1 * Mv1, s2 * Mv2
                # 4) allow swapping the two planes
                for swap in (False, True):
                    a, b = (sv2, sv1) if swap else (sv1, sv2)
                    # 5) sort so that (planeA, planeB) is order‑independent
                    candidate = tuple(sorted([tuple(a), tuple(b)]))
                    candidates.append(candidate)
        # 6) pick the lexicographically smallest fingerprint
        return min(candidates)

    df["canon"] = df.apply(canonical_gb, axis=1)
    df["dupe"] = df.duplicated("canon")
    df_unique = df[~df["dupe"]].copy()
    return df_unique


def _construct_structure_for_entry(args):
    """
    Top-level helper for picklable structure construction.
    """
    from pyiron_workflow_atomistics.gb.gb_code.constructor import (
        construct_GB_from_GBCode,
    )

    (
        entry,
        basis,
        lattice_param,
        equil_volume_per_atom,
        element,
        req_length_grain,
        grain_length_axis,
        min_inplane_gb_length,
    ) = args
    fn = construct_GB_from_GBCode(
        axis=entry["Axis"],
        basis=basis,
        lattice_param=lattice_param,
        m=entry["m"],
        n=entry["n"],
        GB1=entry["GB1"],
        element=element,
        req_length_grain=req_length_grain,
        equil_volume=equil_volume_per_atom,
        grain_length_axis=grain_length_axis,
    )()
    from pyiron_workflow_atomistics.structure_manipulator.tools import (
        create_supercell_with_min_dimensions,
    )

    supercell = create_supercell_with_min_dimensions(
        fn["final_structure"],
        min_dimensions=[min_inplane_gb_length, min_inplane_gb_length, None],
    )()
    # print(type(fn))
    return supercell


def get_gbcode_df(
    axis: np.ndarray,
    basis: str,
    sigma_limit: int,
    lim_plane_index: int,
    max_atoms: int = 100,
    max_workers: int = None,
) -> pd.DataFrame:
    sigma_list, theta_list, m_list, n_list = [], [], [], []

    for i in range(sigma_limit):
        tt = get_theta_m_n_list(axis, i)
        if len(tt) > 0:
            theta, m, n = tt[0]
            if i > 1:
                sigma_list.append(i)
                theta_list.append(theta)
                m_list.append(m)
                n_list.append(n)
            logger.info("Sigma:   {0:3d}  Theta:  {1:5.2f}".format(i, degrees(theta)))
    all_gb_df = pd.DataFrame()

    # 2) Parallel GB gbcode construction parameter df
    all_gb_df = _construct_gbcode_df_for_axis(
        sigma_list=sigma_list,
        theta_list=theta_list,
        m_list=m_list,
        n_list=n_list,
        axis=axis,
        basis=basis,
        lim_plane_index=lim_plane_index,
        max_workers=max_workers,
    )
    all_gb_df = all_gb_df[all_gb_df["n_atoms"] <= max_atoms]
    return all_gb_df


def get_gbcode_df_with_structures(
    df: pd.DataFrame,
    basis: str,
    lattice_param: float,
    equil_volume_per_atom: float,
    element: str = "Fe",
    min_inplane_gb_length: float = 10,
    req_length_grain: float = 15,
    grain_length_axis: float = 0,
    max_workers: int = None,
) -> pd.DataFrame:
    """
    Construct GB structures for each row in the DataFrame in parallel, with a tqdm progress bar.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing GB metadata columns: 'Axis', 'm', 'n', 'GB1'.
    basis : str
        Crystal basis type passed to the constructor.
    lattice_param : float
        Lattice parameter for GB construction.
    equil_volume_per_atom : float
        Equilibrium volume per atom for GB construction.
    element : str, default='Fe'
        Element symbol for GB construction.
    min_inplane_gb_length : float, default=10
        Minimum in-plane GB length parameter.
    req_length_grain : float, default=15
        Required grain length parameter.
    grain_length_axis : float, default=0
        Grain length along the axis.
    max_workers : int or None, optional
        Number of parallel workers; defaults to cpu_count().

    Returns
    -------
    pd.DataFrame
        DataFrame with an added 'structure' column of constructed GBs.
    """
    from tqdm import tqdm

    entries = df.to_dict("records")
    args_list = [
        (
            entry,
            basis,
            lattice_param,
            equil_volume_per_atom,
            element,
            req_length_grain,
            grain_length_axis,
            min_inplane_gb_length,
        )
        for entry in entries
    ]
    n_workers = max_workers if max_workers is not None else cpu_count()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pyiron_workflow_atomistics.gb.gb_code.searcher import (
        _construct_structure_for_entry,
    )

    structures = [None] * len(args_list)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_construct_structure_for_entry, arg): idx
            for idx, arg in enumerate(args_list)
        }
        for future in tqdm(
            as_completed(futures),
            total=len(args_list),
            desc="Constructing GB structures",
        ):
            idx = futures[future]
            structures[idx] = future.result()

    df_out = df.copy()
    df_out["structure"] = structures
    df_out["structure_natoms"] = df_out["structure"].apply(len)
    return df_out


def _rid_negative_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that are redundant due to GB1/GB2 being negations of each other
    with identical number of atoms, grouped by Sigma.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'Sigma', 'GB1', 'GB2', and 'n_atoms' columns.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with one representative per negated GB pair.
    """
    rows_to_drop = set()

    def is_negation(t1, t2):
        return all(x == -y for x, y in zip(t1, t2))

    for sigma, group in tqdm(df.groupby("Sigma"), desc="Removing negative duplicates"):
        processed = set()
        for idx1, row1 in group.iterrows():
            if idx1 in processed:
                continue
            for idx2, row2 in group.iterrows():
                if idx1 != idx2 and idx2 not in processed:
                    if (
                        is_negation(row1["GB1"], row2["GB1"])
                        and is_negation(row1["GB2"], row2["GB2"])
                        and row1["n_atoms"] == row2["n_atoms"]
                    ):

                        rows_to_drop.add(idx2)
                        processed.add(idx1)
                        processed.add(idx2)
                        break  # Remove only one of each mirrored pair

    return df.drop(rows_to_drop)


def _check_duplicates_in_group(group_df: pd.DataFrame) -> list[int]:
    """
    Helper to identify duplicates within a group of structures with the same atom count.
    Returns indices of rows to drop.
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher

    matcher = StructureMatcher()
    unique_structures = []
    rows_to_drop = []
    from pymatgen.io.ase import AseAtomsAdaptor

    for i, row in group_df.iterrows():
        struct = AseAtomsAdaptor().get_structure(row.structure)
        is_duplicate = False
        for u in unique_structures:
            if matcher.fit(struct, u):
                is_duplicate = True
                break
        if is_duplicate:
            rows_to_drop.append(i)
        else:
            unique_structures.append(AseAtomsAdaptor().get_structure(row.structure))

    return rows_to_drop


def _remove_duplicate_structures(
    df: pd.DataFrame, max_atoms: int = 100, max_workers: int = None
) -> pd.DataFrame:
    """
    Removes duplicate structures using structure matching in parallel.
    Groups structures by number of atoms to reduce comparisons.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'structure' column (pymatgen Structure).
    max_atoms : int, optional
        Skip duplicate checking for structures with > max_atoms (default is 100).
    max_workers : int, optional
        Number of processes to use. Defaults to min(cpu_count(), number of groups).

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame.
    """
    df = df.copy()
    df["n_atoms"] = df["structure"].apply(len)
    grouped = [group for _, group in df[df["n_atoms"] <= max_atoms].groupby("n_atoms")]

    if not grouped:
        return df.drop(columns=["n_atoms"])

    if max_workers is None:
        max_workers = min(cpu_count(), len(grouped))

    with Pool(processes=max_workers) as pool:
        results = list(
            tqdm(
                pool.imap(_check_duplicates_in_group, grouped),
                total=len(grouped),
                desc="Duplicate check",
            )
        )

    indices_to_drop = {idx for sublist in results for idx in sublist}
    return df.drop(index=indices_to_drop)  # .drop(columns=["n_atoms"])


def get_gbcode_df_multiple_axes(
    axes_list: list[np.ndarray],
    basis: str = "fcc",
    sigma_limit: int = 100,
    lim_plane_index: int = 3,
    max_atoms: int = 100,
    max_workers: int = None,
) -> pd.DataFrame:
    """
    Parallelized generation of GB DataFrame over multiple axes.
    """
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                get_gbcode_df,
                axis,
                basis,
                sigma_limit,
                lim_plane_index,
                max_atoms,
                max_workers,
            )
            for axis in axes_list
        ]
        all_results = []
        for future in tqdm(futures, desc="Processing all axes"):
            all_results.append(future.result())

    return pd.concat(all_results, ignore_index=True)
