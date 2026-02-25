from __future__ import annotations

import numpy as np

from windturbine.rotor.bem import design_rotor_sections, evaluate_rotor
from windturbine.rotor.polars import PolarModel


def _baseline_sections():
    polar = PolarModel()
    sections = design_rotor_sections(
        radius_m=20.0,
        hub_radius_ratio=0.2,
        n_sections=14,
        blades=3,
        tip_speed_ratio=7.0,
        design_aoa_deg=6.0,
        pitch_deg=0.0,
        polar=polar,
    )
    return polar, sections


def test_chord_and_twist_trend():
    _, sections = _baseline_sections()
    chords = np.array([sec.chord_m for sec in sections], dtype=float)
    twists = np.array([sec.twist_deg for sec in sections], dtype=float)

    assert chords[0] > chords[-1]
    assert np.mean(np.diff(chords)) < 0.0
    assert twists[0] > twists[-1]


def test_bem_performance_sanity():
    polar, sections = _baseline_sections()
    section_results, perf = evaluate_rotor(
        sections=sections,
        blades=3,
        radius_m=20.0,
        hub_radius_ratio=0.2,
        tip_speed_ratio=7.0,
        wind_speed_ms=9.0,
        pitch_deg=0.0,
        air_density=1.225,
        dynamic_viscosity=1.81e-5,
        polar=polar,
    )

    assert len(section_results) == len(sections)
    assert all(sec.reynolds > 0.0 for sec in section_results)
    assert 0.0 < perf.cp < 0.70
    assert 0.0 < perf.ct < 1.50

    converged_fraction = np.mean([1.0 if sec.converged else 0.0 for sec in section_results])
    assert converged_fraction >= 0.7
