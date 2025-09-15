"""Module for calculating and analyzing interstitial positions in crystal structures.

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

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from pyscal3 import Atoms, System


def filter_condition(sys, pos, rvv, distance_min, distance_max, axis, rvv_min, rvv_max):
    ret = False
    # get a midpoint distance;
    mid_point = (distance_min + distance_max) / 2
    width = np.abs(distance_max - mid_point)

    # check if the distance lies within width
    within_distance = distance_min <= pos[axis] <= distance_max

    if not within_distance:
        within_distance = 0 <= np.round(pos[axis], decimals=3) <= width
    if not within_distance:
        within_distance = (
            sys.boxdims[axis] - width
            <= np.round(pos[axis], decimals=3)
            < np.round(sys.boxdims[axis], decimals=3)
        )
    within_rvv = rvv_min < rvv < rvv_max

    if within_distance and within_rvv:
        ret = True
    return ret


def get_ra(sys, natoms, pf):
    """
    Calculate radius ra

    Parameters
    ----------
    sys: pyscal System object

    natoms: int
        total number of atoms in the system

    pf: float
        packing factor of the system

    Returns
    -------
    ra: float
        Calculated ra
    """
    box = sys.box
    vol = np.dot(np.cross(box[0], box[1]), box[2])
    volatom = vol / natoms
    ra = ((pf * volatom) / ((4 / 3) * np.pi)) ** (1 / 3)
    return ra


def get_octahedral_positions(sys_in, alat):
    """
    Get all octahedral vertex positions

    Parameters
    ----------
    sys_in: pyscal System object

    alat: float
        lattice constant in Angstroms

    Returns
    -------
    octahedral_at: list of floats
        position of octahedral voids
    """
    octahedral_at = []
    real_pos = sys_in.atoms.positions
    all_pos = sys_in.atoms["positions"]
    box = sys_in.box
    count = 0
    for i in range(len(all_pos)):
        for j in range(i + 1, len(all_pos)):
            dist = sys_in.calculate.distance(all_pos[i], all_pos[j])
            if np.abs(dist - alat) < 1e-2:
                count += 1
                npos = (np.array(all_pos[i]) + np.array(all_pos[j])) / 2
                if 0 <= npos[0] <= box[0][0]:
                    if 0 <= npos[1] <= box[1][1]:
                        if 0 <= npos[2] <= box[2][2]:
                            # print(np.abs(np.sum(npos-real_pos)))
                            # print(npos)
                            found = False
                            for rpos in real_pos:
                                if np.sum(np.abs(npos - rpos)) < 1e-5:
                                    found = True
                            if not found:
                                octahedral_at.append(npos)
    return octahedral_at


def tabulate_voids(void_ratios, void_count):
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
        if True, write out the system with voids. Only possible in LAMMPS data format.
        Default True
    """
    # read in data
    structure = read(inputfile, format=format)
    sys = System()
    sys.read.file(structure, format="ase")

    # find neighbors
    sys.find.neighbors(method="voronoi", cutoff=0.1)

    # find octahedral voids
    oct = get_octahedral_positions(sys, alat)

    maxcomp = max(list(sys.atoms.composition_ints.keys())) + 1
    ra = get_ra(sys, sys.natoms, pf)

    # concatenate voro voids and octahedral voids
    void_positions = np.concatenate((sys.unique_vertices, oct))
    void_types = [maxcomp for x in range(len(void_positions))]

    # add them to original positions
    all_positions = np.concatenate((sys.atoms.positions, void_positions))
    all_types = np.concatenate((sys.atoms.types, void_types))

    # make new dict with this info
    atom_dict = {"positions": all_positions, "types": all_types}
    atoms = Atoms()
    atoms.from_dict(atom_dict)
    box = sys.box

    # replace system with new info
    sys = System()
    sys.box = box
    sys.atoms = atoms

    # find neighbors again
    sys.find.neighbors(method="cutoff", cutoff=alat)

    # calculate rvv
    rlist = []
    rdist = []
    for count, pos in enumerate(sys.atoms.positions):
        if sys.atoms.types[count] == maxcomp:
            indices = np.array(
                [sys.atoms.types[x] for x in sys.atoms.neighbors.index[count]]
            )
            # print(len(indices))
            args = np.where(indices < maxcomp)
            dists = np.array(sys.atoms.neighbors.distance[count])[args]
            Rvv = min(dists)
            rdist.append(Rvv)
            rvv = (Rvv - ra) / ra
            sys.atoms.neighbors.cutoff[count] = rvv * ra * (1 + extra_rvv)
            rlist.append(rvv)
        else:
            rlist.append(1.0)
            rdist.append(ra)
    void_ratios, void_count = np.unique(np.round(rlist, decimals=3), return_counts=True)
    if tabulate:
        tabulate_voids(void_ratios, void_count)

    sys.write.file(
        f"intermediate_{inputfile}", customkeys=["rvv"], customvals=np.array([rlist]).T
    )

    # use sys as storage for some extra variables
    sys.maxcomp = maxcomp
    sys.rlist = rlist
    return sys, void_ratios, void_count


def filter_and_cluster_atoms(sys, distance, axis, rvv, write=True):
    """
    Filter voids based on conditions

    sys: sys object with previous calculation done
    distance: float of length (2)
        min and max distance cutoff from GB
    axis: int
        axis along which distance is checked
    rvv: float of length (2)
        min and max rvv cutoff
    write: bool, optional
        if True, write final file in LAMMPS format
    """
    d_min = distance[0]
    d_max = distance[1]
    rvv_min = rvv[0]
    rvv_max = rvv[1]
    maxcomp = sys.maxcomp
    rlist = sys.rlist
    conditions = []
    for count, pos in enumerate(sys.atoms.positions):
        if sys.atoms.types[count] == maxcomp:
            condition = filter_condition(
                sys, pos, rlist[count], d_min, d_max, axis, rvv_min, rvv_max
            )
            # add second distance condition check
            # if (rdist[count] < ra):
            #    condition = False
            conditions.append(condition)
        else:
            conditions.append(False)
    sys.find.clusters(conditions, largest=False)
    fdict = {}
    unique_clusters, unique_counts = np.unique(sys.atoms.cluster.id, return_counts=True)
    for count, un in enumerate(unique_clusters):
        if un != -1:
            # if unique_counts[count] > 1:
            args = np.where(sys.atoms.cluster.id == un)[0]
            fdict[str(un)] = args
    mean_positions = []
    mean_rlist = []
    mean_cluster = []
    for key, val in fdict.items():
        if len(val) > 1:
            mean_rlist.append(np.mean([rlist[x] for x in val]))
            pos = [sys.atoms.positions[x] for x in val]
            for i in range(len(pos)):
                for j in range(3):
                    if pos[i][j] > 0.75 * sys.boxdims[j]:
                        pos[i][j] = pos[i][j] - sys.boxdims[j]
            pos = np.mean(pos, axis=0)
            mean_positions.append(pos)
            mean_cluster.append(int(key))
        else:
            mean_positions.append(sys.atoms.positions[val[0]])
            mean_rlist.append(rlist[val[0]])
            mean_cluster.append(int(key))
    previous_positions = []
    previous_types = []

    for count, p in enumerate(sys.atoms.positions):
        if sys.atoms.types[count] < maxcomp:
            previous_positions.append(p)
            previous_types.append(sys.atoms.types[count])
    new_positions = np.concatenate((previous_positions, mean_positions))
    new_rlist = np.concatenate(
        ([1 for x in range(len(previous_positions))], mean_rlist)
    )
    new_types = np.concatenate(
        (previous_types, [maxcomp for x in range(len(mean_positions))])
    )
    new_clusters = np.concatenate(
        ([0 for x in range(len(previous_positions))], mean_cluster)
    )
    atom_dict = {
        "positions": new_positions,
        "types": new_types,
        "rvv": new_rlist,
        "cluster": new_clusters,
    }
    atoms = Atoms()
    atoms.from_dict(atom_dict)
    box = sys.box
    sys = System()
    sys.box = box
    sys.atoms = atoms
    sys = sys.modify.remap_to_box(ghosts=False)
    if write:
        sys.write.file("output.data", customkeys=["rvv", "cluster"])
    return sys
