from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping


def _load_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception:
        data = json.loads(text)
    if not isinstance(data, Mapping):
        raise ValueError(f"Config file must contain a mapping at top level: {path}")
    return data


def _coerce_dataclass(cls: type[Any], raw: Mapping[str, Any] | None) -> Any:
    if raw is None:
        return cls()
    allowed = {field.name for field in fields(cls)}
    kwargs = {key: value for key, value in raw.items() if key in allowed}
    return cls(**kwargs)


@dataclass
class AirfoilConfig:
    radius: float = 0.22648363
    axial_chord: float = 0.05134006
    tangential_chord: float = 0.03934316
    unguided_turning_deg: float = 6.686
    inlet_blade_deg: float = 38.682
    inlet_half_wedge_deg: float = 5.286
    leading_edge_radius: float = 0.00724877
    outlet_blade_deg: float = -76.584
    trailing_edge_radius: float = 0.00246556
    cascade_blade_count: int = 30
    throat: float = 0.00716107
    n_points: int = 80

    def to_turbine_blade_kwargs(self) -> dict[str, Any]:
        import math

        return {
            "radius": self.radius,
            "axial_chord": self.axial_chord,
            "tangential_chord": self.tangential_chord,
            "unguided_turning": math.radians(self.unguided_turning_deg),
            "inlet_blade": math.radians(self.inlet_blade_deg),
            "inlet_half_wedge": math.radians(self.inlet_half_wedge_deg),
            "le_r": self.leading_edge_radius,
            "outlet_blade": math.radians(self.outlet_blade_deg),
            "te_r": self.trailing_edge_radius,
            "n_blades": self.cascade_blade_count,
            "throat": self.throat,
            "n_points": self.n_points,
        }


@dataclass
class RotorConfig:
    radius_m: float = 30.0
    hub_radius_ratio: float = 0.2
    n_sections: int = 18
    wind_speed_ms: float = 9.0
    air_density: float = 1.225
    dynamic_viscosity: float = 1.81e-5
    pitch_deg: float = 0.0


@dataclass
class PolarConfig:
    cl_alpha_per_rad: float = 5.73
    alpha_stall_deg: float = 12.0
    cl_max: float = 1.35
    cd0: float = 0.01
    induced_drag_factor: float = 0.012
    post_stall_drag_gain: float = 0.02


@dataclass
class OptimizerConfig:
    blade_count_candidates: list[int] = None  # type: ignore[assignment]
    aoa_deg_candidates: list[float] = None  # type: ignore[assignment]
    tip_speed_ratio_candidates: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.blade_count_candidates is None:
            self.blade_count_candidates = [2, 3, 4]
        if self.aoa_deg_candidates is None:
            self.aoa_deg_candidates = [4.0, 6.0, 8.0, 10.0]
        if self.tip_speed_ratio_candidates is None:
            self.tip_speed_ratio_candidates = [6.0, 7.0, 8.0]


@dataclass
class SurrogateConfig:
    enabled: bool = True
    n_samples: int = 2000
    degree: int = 2
    seed: int = 42
    top_k_candidates: int = 8
    population_size: int = 120
    generations: int = 45


@dataclass
class OutputConfig:
    directory: str = "outputs"


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class DesignConfig:
    airfoil: AirfoilConfig
    rotor: RotorConfig
    polar: PolarConfig
    optimizer: OptimizerConfig
    surrogate: SurrogateConfig
    output: OutputConfig
    logging: LoggingConfig

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DesignConfig":
        return cls(
            airfoil=_coerce_dataclass(AirfoilConfig, raw.get("airfoil")),
            rotor=_coerce_dataclass(RotorConfig, raw.get("rotor")),
            polar=_coerce_dataclass(PolarConfig, raw.get("polar")),
            optimizer=_coerce_dataclass(OptimizerConfig, raw.get("optimizer")),
            surrogate=_coerce_dataclass(SurrogateConfig, raw.get("surrogate")),
            output=_coerce_dataclass(OutputConfig, raw.get("output")),
            logging=_coerce_dataclass(LoggingConfig, raw.get("logging")),
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "DesignConfig":
        resolved = Path(path)
        raw = _load_mapping(resolved)
        return cls.from_dict(raw)


def load_config(path: Path | str) -> DesignConfig:
    return DesignConfig.from_yaml(path)
