from __future__ import annotations

import builtins
import importlib
import logging
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from windturbine.config import AirfoilConfig

LOGGER = logging.getLogger(__name__)

_TURBINE_FILE_PATH = (
    Path(__file__).resolve().parents[2]
    / "11-Parameters-Turbine-Blade-Generator"
    / "TurbineBladeGen.py"
)
_TURBINE_DIR = _TURBINE_FILE_PATH.parent


@contextmanager
def _temporary_sys_path(path: Path):
    path_str = str(path)
    inserted = False
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass


def _load_turbine_blade_class() -> type[Any]:
    if not _TURBINE_FILE_PATH.exists():
        raise FileNotFoundError(f"Missing TurbineBladeGen.py at expected relative path: {_TURBINE_FILE_PATH}")
    with _temporary_sys_path(_TURBINE_DIR):
        module = importlib.import_module("TurbineBladeGen")
    if not hasattr(module, "TurbineBladeGen"):
        raise AttributeError("TurbineBladeGen not found in imported module")
    return module.TurbineBladeGen


def _pick(params: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in params:
            return params[name]
    raise KeyError(f"Missing parameter. Expected one of: {names}")


def _to_turbine_kwargs(params: Mapping[str, Any], n_points: int) -> dict[str, Any]:
    return {
        "radius": float(_pick(params, "radius")),
        "axial_chord": float(_pick(params, "axial_chord")),
        "tangential_chord": float(_pick(params, "tangential_chord")),
        "unguided_turning": math.radians(float(_pick(params, "unguided_turning_deg", "unguided_turning"))),
        "inlet_blade": math.radians(float(_pick(params, "inlet_blade_deg", "inlet_blade"))),
        "inlet_half_wedge": math.radians(
            float(_pick(params, "inlet_half_wedge_deg", "inlet_half_wedge"))
        ),
        "le_r": float(_pick(params, "leading_edge_radius", "le_r")),
        "outlet_blade": math.radians(float(_pick(params, "outlet_blade_deg", "outlet_blade"))),
        "te_r": float(_pick(params, "trailing_edge_radius", "te_r")),
        "n_blades": int(_pick(params, "cascade_blade_count", "n_blades")),
        "throat": float(_pick(params, "throat")),
        "n_points": int(n_points),
    }


def generate_airfoil_coords(
    params: Mapping[str, Any],
    n_points: int = 200,
    straight_te: bool = True,
) -> pd.DataFrame:
    """Generate pressure/suction/TE points with TurbineBladeGen and save CSV.

    Angles in `params` are expected in degrees and converted to radians internally.
    Saves `outputs/airfoil_coords.csv` and returns DataFrame with columns:
    `surface`, `x`, `y`.
    """
    turbine_kwargs = _to_turbine_kwargs(params=params, n_points=n_points)
    TurbineBladeGen = _load_turbine_blade_class()
    blade = TurbineBladeGen(**turbine_kwargs)

    # Suppress interactive prompt in upstream implementation:
    # "Should data be saved (x and y coordinates; and dependent data)? [y/n]:"
    original_input = builtins.input
    try:
        builtins.input = lambda *args, **kwargs: "n"
        blade.calculate_blade_geometry(straight_te=straight_te)
    finally:
        builtins.input = original_input

    rows: list[dict[str, float | str]] = []
    for x, y in zip(np.asarray(blade.x_pressure), np.asarray(blade.y_pressure)):
        rows.append({"surface": "pressure", "x": float(x), "y": float(y)})
    for x, y in zip(np.asarray(blade.x_suction), np.asarray(blade.y_suction)):
        rows.append({"surface": "suction", "x": float(x), "y": float(y)})
    if blade.x_te_close is not None and blade.y_te_close is not None:
        for x, y in zip(np.asarray(blade.x_te_close), np.asarray(blade.y_te_close)):
            rows.append({"surface": "te", "x": float(x), "y": float(y)})

    df = pd.DataFrame(rows, columns=["surface", "x", "y"])
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "airfoil_coords.csv"
    df.to_csv(output_path, index=False)
    LOGGER.info("Saved airfoil coordinates: %s", output_path.resolve())
    return df


def _smooth_closed_loop(df: pd.DataFrame, n_points: int = 500) -> tuple[np.ndarray, np.ndarray]:
    pressure = df[df["surface"] == "pressure"][["x", "y"]].to_numpy(dtype=float)
    suction = df[df["surface"] == "suction"][["x", "y"]].to_numpy(dtype=float)
    if pressure.size == 0 or suction.size == 0:
        return np.array([]), np.array([])

    pressure = pressure[np.argsort(pressure[:, 0])]
    suction = suction[np.argsort(suction[:, 0])[::-1]]
    loop = np.vstack([pressure, suction, pressure[:1]])

    seg = np.sqrt(np.sum(np.diff(loop, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 1e-12:
        return loop[:, 0], loop[:, 1]
    su = np.linspace(0.0, s[-1], n_points)
    xs = np.interp(su, s, loop[:, 0])
    ys = np.interp(su, s, loop[:, 1])
    return xs, ys


def plot_airfoil(
    df: pd.DataFrame,
    output_path: str | Path = "outputs/airfoil_plot.png",
    aoa_deg: float | None = None,
) -> Path:
    """Plot airfoil coordinates, smooth closed-loop overlay, and optional AoA."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    for surface, color in (("pressure", "tab:blue"), ("suction", "tab:orange"), ("te", "tab:green")):
        subset = df[df["surface"] == surface]
        if not subset.empty:
            plt.plot(subset["x"], subset["y"], ".", ms=3, color=color, label=surface)

    loop_x, loop_y = _smooth_closed_loop(df)
    if loop_x.size > 0:
        plt.plot(loop_x, loop_y, "k-", lw=1.6, label="closed-loop profile")

    if aoa_deg is not None:
        alpha = math.radians(float(aoa_deg))
        x_all = df["x"].to_numpy(dtype=float)
        y_all = df["y"].to_numpy(dtype=float)
        if x_all.size > 1 and y_all.size > 1:
            x0, x1 = float(np.min(x_all)), float(np.max(x_all))
            yc = float(np.mean(y_all))
            L = 0.25 * (x1 - x0)
            # Chord reference
            plt.plot([x0, x0 + L], [yc, yc], "k--", lw=1.0, label="chord ref")
            # Inflow / AoA line
            plt.plot(
                [x0, x0 + L * math.cos(alpha)],
                [yc, yc + L * math.sin(alpha)],
                color="crimson",
                lw=1.4,
                label=f"AoA = {aoa_deg:.2f} deg",
            )
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Pritchard Airfoil Coordinates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    LOGGER.info("Saved airfoil plot: %s", path.resolve())
    return path


@dataclass
class AirfoilCoordinates:
    pressure_x: np.ndarray
    pressure_y: np.ndarray
    suction_x: np.ndarray
    suction_y: np.ndarray
    te_x: np.ndarray
    te_y: np.ndarray

    def iter_rows(self) -> list[tuple[str, float, float]]:
        rows: list[tuple[str, float, float]] = []
        for x, y in zip(self.pressure_x, self.pressure_y):
            rows.append(("pressure", float(x), float(y)))
        for x, y in zip(self.suction_x, self.suction_y):
            rows.append(("suction", float(x), float(y)))
        for x, y in zip(self.te_x, self.te_y):
            rows.append(("te", float(x), float(y)))
        return rows


class PritchardAirfoilGenerator:
    """Compatibility wrapper used by CLI."""

    def generate(self, config: AirfoilConfig, straight_te: bool = True) -> AirfoilCoordinates:
        params = {
            "radius": config.radius,
            "axial_chord": config.axial_chord,
            "tangential_chord": config.tangential_chord,
            "unguided_turning_deg": config.unguided_turning_deg,
            "inlet_blade_deg": config.inlet_blade_deg,
            "inlet_half_wedge_deg": config.inlet_half_wedge_deg,
            "leading_edge_radius": config.leading_edge_radius,
            "outlet_blade_deg": config.outlet_blade_deg,
            "trailing_edge_radius": config.trailing_edge_radius,
            "cascade_blade_count": config.cascade_blade_count,
            "throat": config.throat,
        }
        df = generate_airfoil_coords(params=params, n_points=config.n_points, straight_te=straight_te)
        plot_airfoil(df)

        pressure = df[df["surface"] == "pressure"]
        suction = df[df["surface"] == "suction"]
        te = df[df["surface"] == "te"]
        return AirfoilCoordinates(
            pressure_x=pressure["x"].to_numpy(dtype=float),
            pressure_y=pressure["y"].to_numpy(dtype=float),
            suction_x=suction["x"].to_numpy(dtype=float),
            suction_y=suction["y"].to_numpy(dtype=float),
            te_x=te["x"].to_numpy(dtype=float),
            te_y=te["y"].to_numpy(dtype=float),
        )
