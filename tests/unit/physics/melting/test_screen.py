"""Pure-logic tests for the across-polymorph melting scan (no MD)."""

from pyiron_workflow_atomistics.physics.melting.outputs import PhaseScreenRecord
from pyiron_workflow_atomistics.physics.melting.screen import (
    _dedupe_by_observed,
    _default_candidate_phases,
    _select_for_refinement,
)


def test_default_candidate_phases_dedupes_reference():
    # Reference state already in {fcc,bcc,hcp} -> no duplicate appended.
    assert _default_candidate_phases("Al") == ["fcc", "bcc", "hcp"]  # fcc ref
    assert _default_candidate_phases("Fe") == ["fcc", "bcc", "hcp"]  # bcc ref
    assert _default_candidate_phases("Ti") == ["fcc", "bcc", "hcp"]  # hcp ref


def test_default_candidate_phases_appends_nonstandard_reference():
    # Si reference state is diamond -> appended after the three close-packed phases.
    phases = _default_candidate_phases("Si")
    assert phases[:3] == ["fcc", "bcc", "hcp"]
    assert phases[3] not in ("fcc", "bcc", "hcp")


def test_dedupe_prefers_held_then_higher_tguess():
    # bcc collapsed to fcc (held False) competes with a held fcc on the same
    # observed phase -> the held record wins regardless of t_guess.
    recs = [
        PhaseScreenRecord("bcc", "fcc", 3.2, 999.0, held=False),
        PhaseScreenRecord("fcc", "fcc", 4.0, 900.0, held=True),
    ]
    deduped = _dedupe_by_observed(recs)
    assert len(deduped) == 1
    assert deduped[0].crystalstructure == "fcc"
    assert deduped[0].held is True


def test_dedupe_breaks_ties_by_tguess_when_neither_held():
    recs = [
        PhaseScreenRecord("bcc", "fcc", 3.2, 800.0, held=False),
        PhaseScreenRecord("hcp", "fcc", 2.8, 850.0, held=False),
    ]
    deduped = _dedupe_by_observed(recs)
    assert len(deduped) == 1
    assert deduped[0].t_guess == 850.0


def test_select_takes_top_n_refinable_by_tguess():
    recs = [
        PhaseScreenRecord("fcc", "fcc", 4.0, 900.0, held=True),
        PhaseScreenRecord("bcc", "bcc", 3.2, 950.0, held=True),
        PhaseScreenRecord("hcp", "hcp", 2.8, 600.0, held=True),
    ]
    sel = _select_for_refinement(recs, n_refine=2)
    assert [r.observed_phase for r in sel] == ["bcc", "fcc"]  # top-2 by t_guess


def test_select_filters_non_refinable_phases():
    # 'others'/'ico' cannot be CNA-labelled for solid fraction -> excluded when a
    # refinable phase exists.
    recs = [
        PhaseScreenRecord("fcc", "fcc", 4.0, 900.0, held=True),
        PhaseScreenRecord("diamond", "others", 5.4, 2000.0, held=False),
    ]
    sel = _select_for_refinement(recs, n_refine=2)
    assert [r.observed_phase for r in sel] == ["fcc"]


def test_select_falls_back_when_nothing_refinable():
    recs = [PhaseScreenRecord("diamond", "others", 5.4, 2000.0, held=True)]
    sel = _select_for_refinement(recs, n_refine=2)
    assert len(sel) == 1
    assert sel[0].observed_phase == "others"
