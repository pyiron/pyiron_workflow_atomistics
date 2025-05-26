import numpy as np
import pandas as pd
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import VoronoiNN

def voronoiSiteFeaturiser(atoms: Atoms, site_index: int) -> dict:
    pmg_struct = AseAtomsAdaptor.get_structure(atoms)
    coord_no = VoronoiNN().get_cn(pmg_struct, site_index)
    poly = VoronoiNN().get_voronoi_polyhedra(pmg_struct, site_index)
    volumes   = [poly[k]["volume"]   for k in poly]
    vertices  = [poly[k]["n_verts"]  for k in poly]
    distances = [poly[k]["face_dist"] for k in poly]
    areas     = [poly[k]["area"]     for k in poly]

    stats = lambda arr, name: {
        f"{name}_std":  np.std(arr),
        f"{name}_mean": np.mean(arr),
        f"{name}_min":  np.min(arr),
        f"{name}_max":  np.max(arr),
    }

    out = {
        "VorNN_CoordNo":   coord_no,
        "VorNN_tot_vol":   sum(volumes),
        "VorNN_tot_area":  sum(areas),
    }
    for arr, label in zip([volumes, vertices, areas, distances],
                          ["volumes","vertices","areas","distances"]):
        out.update(stats(arr, f"VorNN_{label}"))
    return out

def distanceMatrixSiteFeaturiser(atoms: Atoms,
                                 site_index: int,
                                 k: int = 6) -> dict:
    """
    Featurise one site by its k nearest neighbor distances
    (using ASE’s full distance matrix with PBC).
    """
    # full NxN distance matrix (MIC = minimum‐image)
    dmat = atoms.get_all_distances(mic=True)
    # distances from this site to all others
    dists = np.delete(dmat[site_index], site_index)
    dists_sorted = np.sort(dists)
    # take exactly k neighbors (pad with NaN if too few)
    if len(dists_sorted) >= k:
        knn = dists_sorted[:k]
    else:
        knn = np.pad(dists_sorted, (0, k - len(dists_sorted)), constant_values=np.nan)

    feats = {f"Dist_knn_{i+1}": float(d) for i, d in enumerate(knn)}
    feats.update({
        "Dist_min":  float(dists_sorted.min()),
        "Dist_mean": float(dists_sorted.mean()),
        "Dist_std":  float(dists_sorted.std()),
        "Dist_max":  float(dists_sorted.max()),
    })
    return feats