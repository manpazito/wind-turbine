from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from windturbine.airfoil.pritchard_generator import (
    AirfoilCoordinates,
    PritchardAirfoilGenerator,
    plot_airfoil,
)
from windturbine.config import load_config
from windturbine.report.justify import build_report_markdown
from windturbine.rotor.bem import SectionResult
from windturbine.rotor.optimizer import optimize_rotor
from windturbine.rotor.polars import PolarModel

LOGGER = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _write_airfoil_csv(path: Path, coords: AirfoilCoordinates) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["surface", "x", "y"])
        for row in coords.iter_rows():
            writer.writerow(row)


def _write_rotor_sections_csv(path: Path, sections: Iterable[SectionResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "r_over_R",
                "r_m",
                "chord_m",
                "twist_deg",
                "aoa_deg",
                "phi_deg",
                "cl",
                "cd",
                "reynolds",
                "local_solidity",
            ]
        )
        for sec in sections:
            writer.writerow(
                [
                    sec.r_over_R,
                    sec.r_m,
                    sec.chord_m,
                    sec.twist_deg,
                    sec.aoa_deg,
                    sec.phi_deg,
                    sec.cl,
                    sec.cd,
                    sec.reynolds,
                    sec.local_solidity,
                ]
            )


def run_design(config_path: Path) -> dict[str, Path]:
    cfg = load_config(config_path)
    _configure_logging(cfg.logging.level)

    output_dir = (Path.cwd() / "outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Generating 2D airfoil profile via TurbineBladeGen wrapper")
    airfoil_generator = PritchardAirfoilGenerator()
    airfoil_coords = airfoil_generator.generate(cfg.airfoil, straight_te=True)

    airfoil_csv = output_dir / "airfoil_coords.csv"
    _write_airfoil_csv(airfoil_csv, airfoil_coords)

    LOGGER.info("Running BEM rotor optimization")
    polar = PolarModel.from_config(cfg.polar)
    result = optimize_rotor(cfg.rotor, cfg.optimizer, polar, cfg.surrogate, output_dir=output_dir)

    rotor_csv = output_dir / "rotor_sections.csv"
    _write_rotor_sections_csv(rotor_csv, result.section_results)
    # Re-render airfoil plot with optimized AoA annotation.
    airfoil_df = pd.read_csv(airfoil_csv)
    plot_airfoil(airfoil_df, output_dir / "airfoil_plot.png", aoa_deg=result.aoa_deg)

    summary = {
        "selected_parameters": {
            "blades": result.blades,
            "design_aoa_deg": result.aoa_deg,
            "tip_speed_ratio": result.tip_speed_ratio,
            "radius_m": cfg.rotor.radius_m,
            "hub_radius_ratio": cfg.rotor.hub_radius_ratio,
            "wind_speed_ms": cfg.rotor.wind_speed_ms,
            "pitch_deg": cfg.rotor.pitch_deg,
        },
        "airfoil_parameters": {
            "radius": cfg.airfoil.radius,
            "axial_chord": cfg.airfoil.axial_chord,
            "tangential_chord": cfg.airfoil.tangential_chord,
            "unguided_turning_deg": cfg.airfoil.unguided_turning_deg,
            "inlet_blade_deg": cfg.airfoil.inlet_blade_deg,
            "outlet_blade_deg": cfg.airfoil.outlet_blade_deg,
        },
        "results": {
            "cp": result.performance.cp,
            "ct": result.performance.ct,
            "power_w": result.performance.power_w,
            "power_kw": result.performance.power_w / 1000.0,
            "thrust_n": result.performance.thrust_n,
            "torque_nm": result.performance.torque_nm,
            "evaluated_candidates": result.evaluated_candidates,
            "section_count": len(result.section_results),
            "converged_sections": sum(1 for sec in result.section_results if sec.converged),
        },
    }

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_md = output_dir / "report.md"
    report_md.write_text(build_report_markdown(cfg, result), encoding="utf-8")

    return {
        "airfoil_csv": airfoil_csv,
        "rotor_csv": rotor_csv,
        "pareto_csv": output_dir / "pareto.csv",
        "surrogate_metrics_json": output_dir / "surrogate_metrics.json",
        "top_design_rotor_sections_csv": output_dir / "top_design_rotor_sections.csv",
        "summary_json": summary_json,
        "report_md": report_md,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wind turbine end-to-end design runner")
    parser.add_argument("--config", required=True, help="Path to config YAML/JSON file")
    args = parser.parse_args(argv)

    outputs = run_design(Path(args.config))
    LOGGER.info("Generated artifacts: %s", {k: str(v) for k, v in outputs.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
