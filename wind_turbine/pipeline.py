from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from wind_turbine.config import Config, load_config
from wind_turbine.optimizer import OptimizationOutcome, run_nsga2, to_dataframe
from wind_turbine.report import build_report
from wind_turbine.xfoil import XfoilPolarDatabase


def _section_dataframe(outcome: OptimizationOutcome, radius_m: float) -> pd.DataFrame:
    best = outcome.best_compromise
    rows: list[dict[str, float]] = []
    for geom, res in zip(best.sections, best.section_results):
        rows.append(
            {
                "r_m": geom.r_m,
                "r_over_R": geom.r_m / radius_m,
                "chord_m": geom.chord_m,
                "twist_deg": geom.twist_deg,
                "phi_deg": res.phi_deg,
                "alpha_deg": res.alpha_deg,
                "reynolds": res.reynolds,
                "cl": res.cl,
                "cd": res.cd,
                "a": res.a,
                "a_prime": res.a_prime,
                "dthrust_n": res.dthrust_n,
                "dtorque_nm": res.dtorque_nm,
                "local_solidity": res.local_solidity,
            }
        )
    return pd.DataFrame(rows)


def _plot_sections(sections_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 4.2))
    axes[0].plot(sections_df["r_over_R"], sections_df["chord_m"], marker="o")
    axes[0].set_xlabel("r/R")
    axes[0].set_ylabel("Chord [m]")
    axes[0].set_title("Chord Distribution")
    axes[0].grid(alpha=0.25)

    axes[1].plot(sections_df["r_over_R"], sections_df["twist_deg"], marker="o", color="tab:orange")
    axes[1].set_xlabel("r/R")
    axes[1].set_ylabel("Twist [deg]")
    axes[1].set_title("Twist Distribution")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_pareto(pareto_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    scatter = ax.scatter(
        pareto_df["root_moment_nm"],
        pareto_df["cp"],
        c=pareto_df["blades"],
        s=40 + 1200.0 * pareto_df["solidity_mean"],
        cmap="viridis",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.4,
    )
    ax.set_xlabel("Root Bending Moment [N.m]")
    ax.set_ylabel("Cp")
    ax.set_title("Pareto Front: Cp vs Root Moment")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Blade count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_airfoil_profile(coords_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.plot(coords_df["x"], coords_df["y"], color="tab:blue", linewidth=1.6)
    ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_pipeline(config: Config) -> dict[str, str]:
    out_dir = config.project.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "xfoil_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    polar_db = XfoilPolarDatabase(config=config.xfoil, cache_dir=cache_dir)
    polar_db.prepare(config.design_space.airfoils)

    outcome = run_nsga2(
        rotor=config.rotor,
        design_space=config.design_space,
        optimizer_cfg=config.optimizer,
        polar_db=polar_db,
        seed=config.project.random_seed,
    )

    all_df = to_dataframe(outcome.all_evaluations)
    pareto_df = to_dataframe(outcome.pareto_front)
    sections_df = _section_dataframe(outcome, radius_m=config.rotor.radius_m)

    all_csv = out_dir / "all_designs.csv"
    pareto_csv = out_dir / "pareto.csv"
    section_csv = out_dir / "best_sections.csv"
    report_md = out_dir / "report.md"
    summary_json = out_dir / "summary.json"
    section_png = out_dir / "best_geometry.png"
    pareto_png = out_dir / "pareto_cp_vs_moment.png"
    airfoil_coords_csv = out_dir / "best_airfoil_coords.csv"
    airfoil_profile_png = out_dir / "best_airfoil_profile.png"

    all_df.to_csv(all_csv, index=False)
    pareto_df.to_csv(pareto_csv, index=False)
    sections_df.to_csv(section_csv, index=False)
    best = outcome.best_compromise
    airfoil_coords_df = polar_db.get_airfoil_coordinates(best.airfoil)
    airfoil_coords_df.to_csv(airfoil_coords_csv, index=False)
    _plot_sections(sections_df, section_png)
    _plot_pareto(pareto_df, pareto_png)
    _plot_airfoil_profile(airfoil_coords_df, airfoil_profile_png, title=f"Airfoil Profile: NACA {best.airfoil}")

    build_report(config=config, outcome=outcome, polar_db=polar_db, output_path=report_md)

    summary = {
        "airfoil": f"NACA {best.airfoil}",
        "blades": best.design.blades,
        "tip_speed_ratio": best.design.tip_speed_ratio,
        "aoa_deg": best.design.aoa_deg,
        "hub_radius_ratio": best.design.hub_radius_ratio,
        "chord_scale": best.design.chord_scale,
        "twist_scale": best.design.twist_scale,
        "cp": best.performance.cp,
        "ct": best.performance.ct,
        "power_w": best.performance.power_w,
        "thrust_n": best.performance.thrust_n,
        "torque_nm": best.performance.torque_nm,
        "root_moment_nm": best.performance.root_moment_nm,
        "solidity_mean": best.performance.solidity_mean,
        "outputs": {
            "all_designs_csv": str(all_csv),
            "pareto_csv": str(pareto_csv),
            "best_sections_csv": str(section_csv),
            "best_airfoil_coords_csv": str(airfoil_coords_csv),
            "report_md": str(report_md),
            "summary_json": str(summary_json),
            "geometry_plot_png": str(section_png),
            "pareto_plot_png": str(pareto_png),
            "best_airfoil_profile_png": str(airfoil_profile_png),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Wind turbine blade optimization using XFOIL + BEM + NSGA-II from scratch."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    summary = run_pipeline(cfg)
    print(json.dumps(summary, indent=2))
    return 0
