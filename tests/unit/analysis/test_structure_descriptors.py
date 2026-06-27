import numpy as np
from ase.build import bulk

from pyiron_workflow_atomistics.analysis.structure_descriptors import (
    analyse_reference_structure,
    classify_solid,
    cna_fractions,
    holes_mask,
    voronoi_max_mean,
)


def _fcc_al():
    return bulk("Al", "fcc", a=4.05, cubic=True).repeat((4, 4, 4))


def test_cna_fractions_fcc_dominant():
    counts = cna_fractions.node_function(_fcc_al())
    assert counts["fcc"] / sum(counts.values()) > 0.95


def test_analyse_reference_structure_fcc():
    key_max, n_atoms, half = analyse_reference_structure.node_function(_fcc_al())
    assert key_max == "fcc"
    assert n_atoms == 256
    assert abs(half - 0.5) < 0.05


def test_classify_solid_true_for_crystal():
    s = _fcc_al()
    key_max, _, half = analyse_reference_structure.node_function(s)
    assert classify_solid.node_function(s, key_max, half) is True


def test_classify_solid_false_for_disordered():
    s = _fcc_al()
    key_max, _, half = analyse_reference_structure.node_function(s)
    rng = np.random.RandomState(0)
    s.set_positions(s.get_positions() + rng.standard_normal((len(s), 3)) * 1.5)
    assert classify_solid.node_function(s, key_max, half) is False


def test_voronoi_max_mean_uniform_fcc():
    vmax, vmean = voronoi_max_mean.node_function(_fcc_al())
    assert vmax / vmean < 1.2  # uniform crystal: max ~ mean


def test_holes_mask_flags_large_void():
    keep = holes_mask.node_function([1.0, 1.0, 5.0], [1.0, 1.0, 1.0], factor=2.0)
    assert keep == [True, True, False]
