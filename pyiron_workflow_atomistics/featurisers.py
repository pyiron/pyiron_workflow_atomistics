import numpy as np
import pandas as pd
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import VoronoiNN


def voronoiSiteFeaturiser(atoms: Atoms, site_index: int) -> dict:
    pmg_struct = AseAtomsAdaptor.get_structure(atoms)
    coord_no = VoronoiNN().get_cn(pmg_struct, site_index)
    poly = VoronoiNN().get_voronoi_polyhedra(pmg_struct, site_index)
    volumes = [poly[k]["volume"] for k in poly]
    vertices = [poly[k]["n_verts"] for k in poly]
    distances = [poly[k]["face_dist"] for k in poly]
    areas = [poly[k]["area"] for k in poly]

    stats = lambda arr, name: {
        f"{name}_std": np.std(arr),
        f"{name}_mean": np.mean(arr),
        f"{name}_min": np.min(arr),
        f"{name}_max": np.max(arr),
    }

    out = {
        "VorNN_CoordNo": coord_no,
        "VorNN_tot_vol": sum(volumes),
        "VorNN_tot_area": sum(areas),
    }
    for arr, label in zip(
        [volumes, vertices, areas, distances],
        ["volumes", "vertices", "areas", "distances"],
    ):
        out.update(stats(arr, f"VorNN_{label}"))
    return out


def distanceMatrixSiteFeaturiser(atoms: Atoms, site_index: int, k: int = 6) -> dict:
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
    feats.update(
        {
            "Dist_min": float(dists_sorted.min()),
            "Dist_mean": float(dists_sorted.mean()),
            "Dist_std": float(dists_sorted.std()),
            "Dist_max": float(dists_sorted.max()),
        }
    )
    return feats


def soapSiteFeaturiser(
    atoms: Atoms,
    site_indices: list[int],
    r_cut: float = 6.0,
    n_max: int = 10,
    l_max: int = 10,
    n_jobs: int = -1,
    periodic: bool = False,
) -> dict:
    try:
        from dscribe.descriptors import SOAP
    except ImportError:
        raise ImportError(
            "dscribe is not installed. Please install it using `pip install dscribe`."
        )
    soap_descriptor = SOAP(
        species=atoms.get_chemical_symbols(),
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        periodic=periodic,
    )
    return soap_descriptor.create(atoms, centers=site_indices, n_jobs=n_jobs)


## Similarity functions
from collections import defaultdict


def summarize_cosine_groups(A, threshold=0.999, ids=None, include_singletons=True):
    """
    Groups rows of A by cosine similarity >= threshold and returns a 2-column DataFrame:
      - 'rep': the first (lowest-index) representative of each group
      - 'same': list of all other IDs in the same group (excluding the representative)

    Parameters
    ----------
    A : array-like, shape (n_samples, n_features)
        Row-wise feature matrix (e.g., SOAP per site).
    threshold : float
        Cosine similarity cutoff to consider rows "the same".
    ids : array-like or None
        Optional external identifiers for rows (e.g., site indices). If None, uses 0..n-1.
    include_singletons : bool
        If True, include rows with no matches as groups with empty 'same' lists.

    Returns
    -------
    pd.DataFrame with columns ['rep', 'same']
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    if ids is None:
        ids = np.arange(n)
    else:
        ids = np.asarray(ids)

    # Normalize rows (cosine = dot of normalized rows)
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X = A / norms

    # Cosine similarity matrix
    S = X @ X.T
    np.fill_diagonal(S, 0.0)  # ignore self

    # Build unions for pairs meeting the threshold
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1

    # Connect edges above threshold (upper triangle only)
    iu = np.triu_indices(n, k=1)
    mask = S[iu] >= threshold
    rows = iu[0][mask]
    cols = iu[1][mask]
    for i, j in zip(rows, cols):
        union(i, j)

    # Gather groups (connected components)
    groups = defaultdict(list)
    for idx in range(n):
        groups[find(idx)].append(idx)

    reps, sames = [], []
    for members in groups.values():
        if not include_singletons and len(members) == 1:
            continue
        members_sorted = sorted(members)
        rep = members_sorted[0]
        same = [m for m in members_sorted if m != rep]
        reps.append(ids[rep])
        sames.append([ids[m] for m in same])

    return pd.DataFrame({"rep": reps, "same": sames})


import numpy as np


def pca_whiten(
    X,
    n_components=0.95,
    method: str = "pca",  # "pca" (default) or "zca"
    model: dict | None = None,
    eps: float = 1e-12,
):
    """
    Center -> PCA -> (optional truncation) -> Whitening.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input feature matrix (e.g., SOAP vectors).
    n_components : float|int
        If float in (0, 1], keep enough PCs to explain this fraction of variance.
        If int >= 1, keep exactly that many PCs (capped at min(n_samples, n_features)).
    method : {"pca","zca"}
        "pca" -> return PCA-whitened scores in k-dim PC space (decorrelated, unit variance).
        "zca" -> return ZCA-whitened data in original feature space (rotated back).
                 If k < d, this is a truncated ZCA using top-k PCs.
    model : dict or None
        If provided, use this fitted model to transform X (no refit).
        Expected keys: {"mu","V","eigvals","k","method","eps"} from a prior fit.
    eps : float
        Small ridge for numerical stability in whitening (adds to eigenvalues).

    Returns
    -------
    Z : np.ndarray
        Whitened data:
          - method="pca": shape (n_samples, k)
          - method="zca": shape (n_samples, n_features)  (truncated if k < d)
    model_out : dict
        Fitted model with keys {"mu","V","eigvals","k","method","eps"}.
        Pass this as `model=` later to transform new data consistently.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    if model is None:
        # center
        mu = X.mean(axis=0)
        Xc = X - mu

        # SVD of centered data: Xc = U S V^T
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        eigvals = (S**2) / max(n - 1, 1)  # eigenvalues of covariance; length = min(n,d)
        V_full = Vt.T  # principal axes (d x r), r=min(n,d)

        # choose k
        if isinstance(n_components, float) and 0 < n_components <= 1:
            cumvar = (
                np.cumsum(eigvals) / np.sum(eigvals)
                if np.sum(eigvals) > 0
                else np.ones_like(eigvals)
            )
            k = int(np.searchsorted(cumvar, n_components) + 1)
        elif isinstance(n_components, int) and n_components >= 1:
            k = min(n_components, V_full.shape[1])
        else:
            raise ValueError("n_components must be float in (0,1] or int >= 1")

        V = V_full[:, :k]  # d x k
        lam = eigvals[:k]  # (k,)
        inv_sqrt = 1.0 / np.sqrt(lam + eps)  # (k,)

        if method.lower() == "pca":
            # PCA scores then scale to unit variance
            Z = (Xc @ V) * inv_sqrt  # (n x k)
        elif method.lower() == "zca":
            # ZCA: project -> scale -> rotate back
            Z = Xc @ (V @ (inv_sqrt[:, None] * V.T))  # (n x d) truncated ZCA if k<d
        else:
            raise ValueError("method must be 'pca' or 'zca'")

        model_out = {
            "mu": mu,
            "V": V,
            "eigvals": eigvals,
            "k": k,
            "method": method.lower(),
            "eps": eps,
        }
        return Z, model_out

    # ---------- transform using existing model ----------
    mu = model["mu"]
    V = model["V"]
    eig_all = model["eigvals"]
    k = model["k"]
    method = model["method"]
    eps = model.get("eps", eps)

    lam = eig_all[:k]
    inv_sqrt = 1.0 / np.sqrt(lam + eps)

    Xc = X - mu
    if method == "pca":
        Z = (Xc @ V) * inv_sqrt
    elif method == "zca":
        Z = Xc @ (V @ (inv_sqrt[:, None] * V.T))
    else:
        raise ValueError("Invalid method in model.")

    return Z, model
