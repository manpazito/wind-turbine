from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from windturbine.config import OptimizerConfig, RotorConfig
from windturbine.rotor.bem import design_and_evaluate
from windturbine.rotor.polars import PolarModel


@dataclass
class DatasetRecord:
    blades: int
    aoa_deg: float
    tip_speed_ratio: float
    cp: float
    ct: float


def generate_training_dataset(
    rotor_cfg: RotorConfig,
    optimizer_cfg: OptimizerConfig,
    polar: PolarModel,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[DatasetRecord]]:
    rng = np.random.default_rng(seed)

    blades_candidates = np.asarray(optimizer_cfg.blade_count_candidates, dtype=int)
    aoa_min = float(min(optimizer_cfg.aoa_deg_candidates))
    aoa_max = float(max(optimizer_cfg.aoa_deg_candidates))
    tsr_min = float(min(optimizer_cfg.tip_speed_ratio_candidates))
    tsr_max = float(max(optimizer_cfg.tip_speed_ratio_candidates))

    features: list[list[float]] = []
    targets: list[float] = []
    records: list[DatasetRecord] = []

    for _ in range(n_samples):
        blades = int(rng.choice(blades_candidates))
        aoa_deg = float(rng.uniform(aoa_min, aoa_max))
        tsr = float(rng.uniform(tsr_min, tsr_max))

        _, _, performance = design_and_evaluate(
            radius_m=rotor_cfg.radius_m,
            hub_radius_ratio=rotor_cfg.hub_radius_ratio,
            n_sections=rotor_cfg.n_sections,
            blades=blades,
            tip_speed_ratio=tsr,
            design_aoa_deg=aoa_deg,
            wind_speed_ms=rotor_cfg.wind_speed_ms,
            pitch_deg=rotor_cfg.pitch_deg,
            air_density=rotor_cfg.air_density,
            dynamic_viscosity=rotor_cfg.dynamic_viscosity,
            polar=polar,
        )

        if not np.isfinite(performance.cp):
            continue

        features.append([float(blades), aoa_deg, tsr])
        targets.append(float(performance.cp))
        records.append(
            DatasetRecord(
                blades=blades,
                aoa_deg=aoa_deg,
                tip_speed_ratio=tsr,
                cp=float(performance.cp),
                ct=float(performance.ct),
            )
        )

    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    return x, y, records
