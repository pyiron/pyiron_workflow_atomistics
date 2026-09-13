"""Point-defect builders: vacancies, substitutionals, interstitial-site finders.

The interstitial / void analysis utilities are adapted from:

Authors
-------
Abril Azocar Guzman (ORCiD: 0000-0001-7564-7990)
Rebecca Janisch (ORCiD: 0000-0003-2136-0788)

Reference
---------
Azócar Guzmán, A., & Janisch, R. (2024). Effects of mechanical stress, chemical potential,
and coverage on hydrogen solubility during hydrogen-enhanced decohesion of ferritic steel
grain boundaries: A first-principles study. Phys. Rev. Mater., 8, 073601.
doi:10.1103/PhysRevMaterials.8.073601
"""

import itertools

import flowrep as fr
import numpy as np
from ase import Atoms as ASEAtoms

#: Species used to mark a void site. ASE's placeholder element (Z = 0).
VOID_SYMBOL = "X"


@fr.atomic("vacancy_structure")
def create_vacancy(structure: ASEAtoms, remove_atom_index: int = 0) -> ASEAtoms:
    """Return a copy of ``structure`` with one atom removed.

    Examples
    --------
    >>> from ase.build import bulk
    >>> bulk_struct = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    >>> vac = create_vacancy(bulk_struct, remove_atom_index=0)
    >>> len(vac) == len(bulk_struct) - 1
    True
    """
    vacancy_structure = structure.copy()
    vacancy_structure.pop(remove_atom_index)
    return vacancy_structure


@fr.atomic("structure")
def substitutional_swap(
    base_structure: ASEAtoms, defect_site: int = 0, new_symbol: str = "Si"
) -> ASEAtoms:
    """Return a copy of ``base_structure`` with one atom's symbol swapped."""
    structure = base_structure.copy()
    structure[defect_site].symbol = new_symbol
    return structure


def _void_mask(structure: ASEAtoms) -> np.ndarray:
    """Boolean mask of the void sites (placeholder species) in a structure."""
    return np.asarray(structure.numbers) == 0


def _neighbor_data(structure: ASEAtoms, key: str):
    """Read a per-atom neighbour array, which pyscal3 stores in ``arrays``
    when every atom has the same number of neighbours and in ``info``
    otherwise."""
    if key in structure.arrays:
        return structure.arrays[key]
    return structure.info[key]


def filter_condition(
    structure, pos, rvv, distance_min, distance_max, axis, rvv_min, rvv_max
):
    """
    Check whether a void at ``pos`` falls inside the requested distance band
    and void-ratio range.

    Parameters
    ----------
    structure: ase.Atoms
        the structure the void belongs to, used for the cell lengths
    """
    ret = False
    # get a midpoint distance;
    mid_point = (distance_min + distance_max) / 2
    width = np.abs(distance_max - mid_point)

    box_lengths = np.asarray(structure.cell.lengths(), dtype=float)

    # check if the distance lies within width
    within_distance = distance_min <= pos[axis] <= distance_max

    if not within_distance:
        within_distance = 0 <= np.round(pos[axis], decimals=3) <= width
    if not within_distance:
        within_distance = (
            box_lengths[axis] - width
            <= np.round(pos[axis], decimals=3)
            < np.round(box_lengths[axis], decimals=3)
        )
    within_rvv = rvv_min < rvv < rvv_max

    if within_distance and within_rvv:
        ret = True
    return ret


def get_ra(structure, natoms, pf):
    """
    Calculate radius ra

    Parameters
    ----------
    structure: ase.Atoms

    natoms: int
        total number of atoms in the system

    pf: float
        packing factor of the system

    Returns
    -------
    ra: float
        Calculated ra
    """
    vol = abs(np.linalg.det(np.asarray(structure.cell)))
    volatom = vol / natoms
    ra = ((pf * volatom) / ((4 / 3) * np.pi)) ** (1 / 3)
    return ra


def get_octahedral_positions(structure, alat, tolerance=1e-2):
    """
    Get all octahedral vertex positions

    Every pair of atoms separated by ``alat`` defines an octahedral site at
    its midpoint. Periodic images are included so that sites at the cell
    boundary are found as well; midpoints outside the cell, and those that
    coincide with an existing atom, are discarded.

    Parameters
    ----------
    structure: ase.Atoms

    alat: float
        lattice constant in Angstroms

    tolerance: float, optional
        how far a pair distance may deviate from ``alat``. Default 1e-2

    Returns
    -------
    octahedral_at: list of floats
        position of octahedral voids
    """
    from scipy.spatial import cKDTree

    cell = np.asarray(structure.cell, dtype=float)
    real_pos = np.asarray(structure.positions, dtype=float)

    # periodic images, the ghost atoms the pyscal System used to carry
    offsets = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float)
    all_pos = (real_pos[None, :, :] + (offsets @ cell)[:, None, :]).reshape(-1, 3)

    tree = cKDTree(all_pos)
    pairs = tree.query_pairs(alat + tolerance, output_type="ndarray")
    if len(pairs) == 0:
        return []
    dist = np.linalg.norm(all_pos[pairs[:, 0]] - all_pos[pairs[:, 1]], axis=1)
    pairs = pairs[np.abs(dist - alat) < tolerance]

    midpoints = 0.5 * (all_pos[pairs[:, 0]] + all_pos[pairs[:, 1]])
    bounds = np.array([cell[0][0], cell[1][1], cell[2][2]], dtype=float)
    inside = np.all((midpoints >= 0) & (midpoints <= bounds), axis=1)
    midpoints = midpoints[inside]

    octahedral_at = []
    for npos in midpoints:
        # drop midpoints sitting on top of an existing atom
        if np.min(np.sum(np.abs(npos - real_pos), axis=1)) < 1e-5:
            continue
        octahedral_at.append(npos)
    return octahedral_at


def tabulate_voids(void_ratios, void_count):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3, 2))

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    collabel = ("Value", "Count")
    y = ax.table(
        cellText=np.array([void_ratios, void_count]).T, colLabels=collabel, loc="center"
    )
    y.set_fontsize(14)
    y.scale(2, 2)


def calculate_voids(
    inputfile, format, alat, pf, extra_rvv=0.5, tabulate=True, write=True
):
    """
    Read in a file and calculate voids

    Parameters
    ----------
    inputfile: string
        input file with data

    format: string
        format of the input file

    alat: float
        lattice constant in angstrom

    pf: float
        packing fraction

    extra_rvv: float, optional
        a safe factor used in clustering to ensure overlapping atoms
        are removed. Default 0.5

    tabulate: bool, optional
        if True, plot a table. Default True

    write: bool, optional
        if True, write the structure with the voids to
        ``intermediate_<inputfile>.xyz``. Default True

    Returns
    -------
    structure: ase.Atoms
        the atoms of the input file followed by the void sites, which carry
        the placeholder species ``X``. The void ratio of every site is in
        ``structure.arrays["rvv"]``.

    void_ratios, void_count: numpy.ndarray
        the distinct void ratios and how often each occurs
    """
    import pyscal3
    from ase.io import read

    # read in data
    structure = read(inputfile, format=format)

    # find neighbors
    pyscal3.find_neighbors(structure, method="voronoi", cutoff=0.1)
    voronoi_vertices = np.asarray(structure.info["pyscal_unique_vertices"])

    # find octahedral voids
    oct = get_octahedral_positions(structure, alat)

    ra = get_ra(structure, len(structure), pf)

    # concatenate voro voids and octahedral voids
    void_positions = np.concatenate((voronoi_vertices, oct))

    # a clean copy without the pyscal3 results, so the two can be joined
    real = ASEAtoms(
        numbers=structure.numbers,
        positions=structure.positions,
        cell=structure.cell,
        pbc=structure.pbc,
    )
    voids = ASEAtoms(
        symbols=[VOID_SYMBOL] * len(void_positions),
        positions=void_positions,
        cell=structure.cell,
        pbc=structure.pbc,
    )
    combined = real + voids
    is_void = _void_mask(combined)

    # find neighbors again
    pyscal3.find_neighbors(combined, method="cutoff", cutoff=alat)
    neighbors = _neighbor_data(combined, "pyscal_neighbors")
    neighbordist = _neighbor_data(combined, "pyscal_neighbordist")
    cutoffs = np.asarray(combined.arrays["pyscal_cutoff"], dtype=float)

    # calculate rvv
    rlist = np.ones(len(combined), dtype=float)
    for count in np.flatnonzero(is_void):
        index = np.asarray(neighbors[count], dtype=int)
        dists = np.asarray(neighbordist[count], dtype=float)
        real_neighbors = index[~is_void[index]] if len(index) else index
        if len(real_neighbors) == 0:
            continue
        Rvv = np.min(dists[~is_void[index]])
        rvv = (Rvv - ra) / ra
        # the clustering below groups voids that lie within this distance
        cutoffs[count] = rvv * ra * (1 + extra_rvv)
        rlist[count] = rvv
    combined.arrays["pyscal_cutoff"] = cutoffs
    combined.arrays["rvv"] = rlist

    void_ratios, void_count = np.unique(np.round(rlist, decimals=3), return_counts=True)
    if tabulate:
        tabulate_voids(void_ratios, void_count)

    if write:
        _write_with_arrays(f"intermediate_{inputfile}.xyz", combined, ["rvv"])

    return combined, void_ratios, void_count


def _write_with_arrays(filename, structure, keys):
    """Write a structure with extra per-atom columns.

    pyscal3 4.0 has no file writer, so the extended XYZ format is used: it
    keeps the cell and any per-atom array, which the LAMMPS data format
    cannot.
    """
    from ase.io import write as ase_write

    out = ASEAtoms(
        numbers=structure.numbers,
        positions=structure.positions,
        cell=structure.cell,
        pbc=structure.pbc,
    )
    for key in keys:
        out.arrays[key] = np.asarray(structure.arrays[key])
    ase_write(filename, out, format="extxyz")


def filter_and_cluster_atoms(structure, distance, axis, rvv, write=True):
    """
    Filter voids based on conditions

    structure: ase.Atoms
        the structure returned by :func:`calculate_voids`
    distance: float of length (2)
        min and max distance cutoff from GB
    axis: int
        axis along which distance is checked
    rvv: float of length (2)
        min and max rvv cutoff
    write: bool, optional
        if True, write the final structure to ``output.xyz``

    Returns
    -------
    ase.Atoms
        the real atoms followed by one site per void cluster, with the void
        ratio in ``arrays["rvv"]`` and the cluster id in ``arrays["cluster"]``
    """
    import pyscal3

    d_min = distance[0]
    d_max = distance[1]
    rvv_min = rvv[0]
    rvv_max = rvv[1]
    is_void = _void_mask(structure)
    rlist = np.asarray(structure.arrays["rvv"], dtype=float)
    box_lengths = np.asarray(structure.cell.lengths(), dtype=float)

    conditions = []
    for count, pos in enumerate(structure.positions):
        if is_void[count]:
            condition = filter_condition(
                structure, pos, rlist[count], d_min, d_max, axis, rvv_min, rvv_max
            )
            # add second distance condition check
            # if (rdist[count] < ra):
            #    condition = False
            conditions.append(condition)
        else:
            conditions.append(False)

    pyscal3.find_clusters(structure, conditions, largest=False)
    cluster_ids = np.asarray(structure.arrays["pyscal_cluster"])

    fdict = {}
    unique_clusters = np.unique(cluster_ids)
    for un in unique_clusters:
        if un != -1:
            args = np.where(cluster_ids == un)[0]
            fdict[str(un)] = args

    mean_positions = []
    mean_rlist = []
    mean_cluster = []
    for key, val in fdict.items():
        if len(val) > 1:
            mean_rlist.append(np.mean([rlist[x] for x in val]))
            pos = [np.array(structure.positions[x], dtype=float) for x in val]
            for i in range(len(pos)):
                for j in range(3):
                    if pos[i][j] > 0.75 * box_lengths[j]:
                        pos[i][j] = pos[i][j] - box_lengths[j]
            pos = np.mean(pos, axis=0)
            mean_positions.append(pos)
            mean_cluster.append(int(key))
        else:
            mean_positions.append(structure.positions[val[0]])
            mean_rlist.append(rlist[val[0]])
            mean_cluster.append(int(key))

    previous_positions = structure.positions[~is_void]
    previous_numbers = np.asarray(structure.numbers)[~is_void]

    if len(mean_positions) == 0:
        mean_positions = np.zeros((0, 3))
    new_positions = np.concatenate((previous_positions, mean_positions))
    new_rlist = np.concatenate(
        (np.ones(len(previous_positions)), mean_rlist)
    )
    new_numbers = np.concatenate(
        (previous_numbers, np.zeros(len(mean_positions), dtype=int))
    )
    new_clusters = np.concatenate(
        (np.zeros(len(previous_positions), dtype=int), mean_cluster)
    )

    out = ASEAtoms(
        numbers=new_numbers,
        positions=new_positions,
        cell=structure.cell,
        pbc=True,
    )
    out.arrays["rvv"] = np.asarray(new_rlist, dtype=float)
    out.arrays["cluster"] = np.asarray(new_clusters, dtype=int)
    out.wrap()
    if write:
        _write_with_arrays("output.xyz", out, ["rvv", "cluster"])
    return out
