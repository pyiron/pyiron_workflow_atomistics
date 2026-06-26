from __future__ import annotations

import numpy as np
import pyiron_workflow as pwf


@pwf.as_function_node("strains", "ratios", "pressures", "temperatures", "sl_flag")
def ratio_selection(strains, ratios, pressures, temperatures, ratio_boundary: float = 0.25):
    """Keep the longest contiguous strain window with ratio in 0.5 +/- boundary.

    ``sl_flag`` is +1 if the selected window is mostly solid (>0.5), else -1.
    """
    groups, current = [], []
    for r in ratios:
        if (0.5 - ratio_boundary) < r < (0.5 + ratio_boundary):
            current.append(r)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    if not groups:
        sel_strains, sel_ratios, sel_pressures, sel_temperatures = [], [], [], []
        flag = 1 if np.mean(ratios) > 0.5 else -1
    else:
        best = groups[int(np.argmax([len(g) for g in groups]))]
        keep = [r in best for r in ratios]
        sel_r = np.array(ratios)[keep]
        flag = 1 if np.mean(sel_r) > 0.5 else -1
        sel_strains = np.array(strains)[keep].tolist()
        sel_ratios = sel_r.tolist()
        sel_pressures = np.array(pressures)[keep].tolist()
        sel_temperatures = np.array(temperatures)[keep].tolist()
    return sel_strains, sel_ratios, sel_pressures, sel_temperatures, flag


@pwf.as_function_node("t_next", "t_mean", "t_left", "t_right")
def predict_melting_point(strains, pressures, temperatures, boundary_value: float = 0.25):
    """Extrapolate temperature to zero pressure; bracket via boundary_value."""
    fit_temp_from_press = np.poly1d(np.polyfit(pressures, temperatures, 1))
    t_next = float(fit_temp_from_press(0.0))
    t_min, t_max = float(np.min(temperatures)), float(np.max(temperatures))
    span = t_max - t_min
    t_mean = t_min + span * 0.5
    t_left = t_min + span * (0.5 - boundary_value)
    t_right = t_min + span * (0.5 + boundary_value)
    return t_next, t_mean, t_left, t_right
