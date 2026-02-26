from __future__ import annotations

from wind_turbine.config import DesignSpaceConfig, OptimizerConfig, RotorConfig
from wind_turbine.optimizer import run_nsga2
from wind_turbine.xfoil import PolarPoint


class DummyPolars:
    def sample(self, airfoil: str, reynolds: float, alpha_deg: float) -> PolarPoint:
        # Mildly airfoil-dependent so optimizer sees differences.
        af_factor = 0.0 if airfoil == "4412" else 0.04
        cl = 0.85 + af_factor + 0.025 * (alpha_deg - 6.0)
        cd = 0.012 + 0.0006 * (alpha_deg - 6.0) ** 2 + 0.0002 * af_factor
        return PolarPoint(cl=cl, cd=cd)


def test_nsga2_returns_nonempty_pareto() -> None:
    rotor = RotorConfig(radius_m=12.0, wind_speed_ms=7.5, n_sections=10)
    design_space = DesignSpaceConfig(
        airfoils=["4412", "2412"],
        blades_options=[2, 3, 4],
        tip_speed_ratio_range=(5.0, 8.5),
        aoa_deg_range=(4.0, 9.0),
        hub_radius_ratio_range=(0.16, 0.25),
        chord_scale_range=(0.85, 1.2),
        twist_scale_range=(0.9, 1.1),
        chord_ratio_limits=(0.02, 0.12),
    )
    opt = OptimizerConfig(population_size=14, generations=3, crossover_probability=0.9, mutation_probability=0.3)
    outcome = run_nsga2(
        rotor=rotor,
        design_space=design_space,
        optimizer_cfg=opt,
        polar_db=DummyPolars(),  # type: ignore[arg-type]
        seed=7,
    )
    assert len(outcome.pareto_front) >= 1
    assert outcome.best_compromise.performance.cp > -1.0
