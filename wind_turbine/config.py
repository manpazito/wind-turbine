from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    output_dir: Path = Path("outputs")
    random_seed: int = 42


@dataclass(frozen=True)
class RotorConfig:
    radius_m: float = 40.0
    wind_speed_ms: float = 9.0
    air_density: float = 1.225
    dynamic_viscosity: float = 1.81e-5
    pitch_deg: float = 0.0
    n_sections: int = 18


@dataclass(frozen=True)
class DesignSpaceConfig:
    airfoils: list[str] = field(default_factory=lambda: ["4412", "4415", "2412", "0012", "23012"])
    blades_options: list[int] = field(default_factory=lambda: [2, 3, 4, 5])
    tip_speed_ratio_range: tuple[float, float] = (4.5, 10.0)
    aoa_deg_range: tuple[float, float] = (3.0, 11.0)
    hub_radius_ratio_range: tuple[float, float] = (0.15, 0.28)
    chord_scale_range: tuple[float, float] = (0.75, 1.35)
    twist_scale_range: tuple[float, float] = (0.85, 1.20)
    chord_ratio_limits: tuple[float, float] = (0.015, 0.12)


@dataclass(frozen=True)
class OptimizerConfig:
    population_size: int = 48
    generations: int = 18
    crossover_probability: float = 0.9
    mutation_probability: float = 0.25


@dataclass(frozen=True)
class XfoilConfig:
    executable: str = "xfoil"
    reynolds_bins: list[float] = field(
        default_factory=lambda: [150000, 300000, 600000, 1000000, 1800000, 3000000]
    )
    alpha_start_deg: float = -6.0
    alpha_end_deg: float = 16.0
    alpha_step_deg: float = 0.5
    ncrit: float = 9.0
    max_iter: int = 150
    timeout_s: float = 25.0


@dataclass(frozen=True)
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    rotor: RotorConfig = field(default_factory=RotorConfig)
    design_space: DesignSpaceConfig = field(default_factory=DesignSpaceConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    xfoil: XfoilConfig = field(default_factory=XfoilConfig)


def _to_tuple2(raw: Any, name: str) -> tuple[float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"`{name}` must be a list of length 2.")
    lo = float(raw[0])
    hi = float(raw[1])
    if lo >= hi:
        raise ValueError(f"`{name}` lower bound must be less than upper bound.")
    return lo, hi


def _merge_dataclass(dc_type: Any, raw: dict[str, Any] | None) -> Any:
    if raw is None:
        return dc_type()
    kwargs: dict[str, Any] = {}
    for key in dc_type.__dataclass_fields__.keys():
        if key in raw:
            kwargs[key] = raw[key]
    return dc_type(**kwargs)


def load_config(path: str | Path) -> Config:
    cfg_path = Path(path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    project_raw = data.get("project", {})
    rotor_raw = data.get("rotor", {})
    design_raw = data.get("design_space", {})
    optimizer_raw = data.get("optimizer", {})
    xfoil_raw = data.get("xfoil", {})

    project = _merge_dataclass(ProjectConfig, project_raw)
    rotor = _merge_dataclass(RotorConfig, rotor_raw)
    optimizer = _merge_dataclass(OptimizerConfig, optimizer_raw)
    xfoil = _merge_dataclass(XfoilConfig, xfoil_raw)
    design_space = _merge_dataclass(DesignSpaceConfig, design_raw)

    # Normalize and validate range-like values.
    design_space = DesignSpaceConfig(
        airfoils=[str(a).strip() for a in design_space.airfoils],
        blades_options=[int(b) for b in design_space.blades_options],
        tip_speed_ratio_range=_to_tuple2(design_space.tip_speed_ratio_range, "tip_speed_ratio_range"),
        aoa_deg_range=_to_tuple2(design_space.aoa_deg_range, "aoa_deg_range"),
        hub_radius_ratio_range=_to_tuple2(design_space.hub_radius_ratio_range, "hub_radius_ratio_range"),
        chord_scale_range=_to_tuple2(design_space.chord_scale_range, "chord_scale_range"),
        twist_scale_range=_to_tuple2(design_space.twist_scale_range, "twist_scale_range"),
        chord_ratio_limits=_to_tuple2(design_space.chord_ratio_limits, "chord_ratio_limits"),
    )

    if not design_space.airfoils:
        raise ValueError("At least one airfoil must be provided.")
    if not design_space.blades_options:
        raise ValueError("At least one blade count option must be provided.")
    if any(b < 1 for b in design_space.blades_options):
        raise ValueError("All blade counts must be >= 1.")
    if rotor.n_sections < 4:
        raise ValueError("`rotor.n_sections` must be >= 4.")
    if rotor.radius_m <= 0.0:
        raise ValueError("`rotor.radius_m` must be > 0.")
    if optimizer.population_size < 8:
        raise ValueError("`optimizer.population_size` must be >= 8.")
    if optimizer.generations < 1:
        raise ValueError("`optimizer.generations` must be >= 1.")
    if not xfoil.reynolds_bins:
        raise ValueError("`xfoil.reynolds_bins` must not be empty.")
    if xfoil.alpha_step_deg <= 0:
        raise ValueError("`xfoil.alpha_step_deg` must be > 0.")

    out_dir = Path(project.output_dir)
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()

    project = ProjectConfig(output_dir=out_dir, random_seed=int(project.random_seed))
    return Config(project=project, rotor=rotor, design_space=design_space, optimizer=optimizer, xfoil=xfoil)
