import numpy as np
from ase.build import bulk

from pyiron_workflow_atomistics.physics.melting.solid_fraction import solid_fraction_kde


def _half_solid_half_liquid():
    s = bulk("Al", "fcc", a=4.05, cubic=True).repeat((4, 4, 8))  # long in z
    z = s.get_scaled_positions()[:, 2]
    upper = np.where(z >= 0.5)[0]
    rng = np.random.RandomState(0)
    pos = s.get_positions()
    pos[upper] += rng.standard_normal((len(upper), 3)) * 1.6  # melt upper half
    s.set_positions(pos)
    return s


def test_full_solid_ratio_near_one():
    s = bulk("Al", "fcc", a=4.05, cubic=True).repeat((4, 4, 8))
    assert solid_fraction_kde.node_function(s, "fcc") > 0.9


def test_half_solid_ratio_near_half():
    frac = solid_fraction_kde.node_function(_half_solid_half_liquid(), "fcc")
    assert 0.3 < frac < 0.7


def test_all_liquid_ratio_zero():
    s = bulk("Al", "fcc", a=4.05, cubic=True).repeat((4, 4, 8))
    rng = np.random.RandomState(1)
    s.set_positions(s.get_positions() + rng.standard_normal((len(s), 3)) * 2.0)
    assert solid_fraction_kde.node_function(s, "fcc") == 0.0
