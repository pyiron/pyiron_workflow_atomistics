from pyiron_workflow_atomistics.physics.melting.fitting import (
    predict_melting_point,
    ratio_selection,
)


def test_ratio_selection_picks_coexistence_window():
    strains = [0.96, 0.98, 1.00, 1.02, 1.04]
    ratios = [0.05, 0.45, 0.50, 0.55, 0.95]  # middle three near 0.5
    pressures = [2.0, 1.0, 0.0, -1.0, -2.0]
    temps = [900, 905, 910, 915, 920]
    ss, rr, pp, tt, flag = ratio_selection.node_function(
        strains, ratios, pressures, temps, ratio_boundary=0.25
    )
    assert list(ss) == [0.98, 1.00, 1.02]
    assert flag in (1, -1)


def test_predict_melting_point_extrapolates_to_zero_pressure():
    # T = 910 + 250*P  => T(P=0) = 910; P linear in strain
    strains = [0.98, 1.00, 1.02]
    pressures = [1.0, 0.0, -1.0]
    temps = [910 + 250 * p for p in pressures]
    t_next, t_mean, t_left, t_right = predict_melting_point.node_function(
        strains, pressures, temps, boundary_value=0.25
    )
    assert abs(t_next - 910.0) < 1e-6
    assert t_left < t_mean < t_right
