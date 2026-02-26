from __future__ import annotations

from wind_turbine.bem import design_blade_geometry, evaluate_rotor
from wind_turbine.config import DesignSpaceConfig, RotorConfig
from wind_turbine.xfoil import PolarPoint


class DummyPolars:
    def sample(self, airfoil: str, reynolds: float, alpha_deg: float) -> PolarPoint:
        cl = 0.85 + 0.03 * (alpha_deg - 6.0)
        cd = 0.012 + 0.0005 * (alpha_deg - 6.0) ** 2
        return PolarPoint(cl=cl, cd=cd)


def test_bem_design_and_eval_are_physical() -> None:
    rotor = RotorConfig(radius_m=20.0, wind_speed_ms=8.5, n_sections=14)
    design_space = DesignSpaceConfig()
    polars = DummyPolars()

    sections = design_blade_geometry(
        rotor=rotor,
        design_space=design_space,
        polar_db=polars,  # type: ignore[arg-type]
        airfoil="4412",
        blades=3,
        tip_speed_ratio=7.0,
        design_aoa_deg=6.0,
        hub_radius_ratio=0.20,
        chord_scale=1.0,
        twist_scale=1.0,
    )
    section_results, performance = evaluate_rotor(
        rotor=rotor,
        polar_db=polars,  # type: ignore[arg-type]
        airfoil="4412",
        blades=3,
        tip_speed_ratio=7.0,
        hub_radius_ratio=0.20,
        sections=sections,
    )

    assert len(section_results) == rotor.n_sections
    assert performance.cp > -1.0
    assert performance.thrust_n > 0.0
    assert sections[0].twist_deg > sections[-1].twist_deg
