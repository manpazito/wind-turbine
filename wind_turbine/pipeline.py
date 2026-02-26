from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wind_turbine.config import Config, load_config
from wind_turbine.optimizer import (
    DesignVector,
    OptimizationOutcome,
    evaluate_design,
    run_nsga2,
    to_dataframe,
)
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


def _objective_vector_from_eval(cp: float, root_moment_nm: float, solidity_mean: float) -> np.ndarray:
    return np.array([-cp, root_moment_nm, solidity_mean], dtype=float)


def _objective_distance(obj: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> float:
    norm = (obj - ideal) / (nadir - ideal + 1e-12)
    return float(np.linalg.norm(norm))


def _compute_local_sensitivity(
    config: Config,
    outcome: OptimizationOutcome,
    polar_db: XfoilPolarDatabase,
) -> pd.DataFrame:
    base = outcome.best_compromise
    base_design = base.design
    base_obj = _objective_vector_from_eval(
        cp=base.performance.cp,
        root_moment_nm=base.performance.root_moment_nm,
        solidity_mean=base.performance.solidity_mean,
    )

    pareto_objs = np.array([np.array(p.objectives, dtype=float) for p in outcome.pareto_front], dtype=float)
    ideal = pareto_objs.min(axis=0)
    nadir = pareto_objs.max(axis=0)
    base_dist = _objective_distance(base_obj, ideal, nadir)

    eval_cache: dict[tuple[int, int, float, float, float, float, float], object] = {}

    def get_eval(design: DesignVector):
        key = design.cache_key()
        if key not in eval_cache:
            eval_cache[key] = evaluate_design(
                design=design,
                rotor=config.rotor,
                design_space=config.design_space,
                polar_db=polar_db,
            )
        return eval_cache[key]

    def dist_from_eval(e) -> float:
        obj = _objective_vector_from_eval(
            cp=e.performance.cp,
            root_moment_nm=e.performance.root_moment_nm,
            solidity_mean=e.performance.solidity_mean,
        )
        return _objective_distance(obj, ideal, nadir)

    rows: list[dict[str, object]] = []
    continuous_specs = [
        ("tip_speed_ratio", config.design_space.tip_speed_ratio_range),
        ("aoa_deg", config.design_space.aoa_deg_range),
        ("hub_radius_ratio", config.design_space.hub_radius_ratio_range),
        ("chord_scale", config.design_space.chord_scale_range),
        ("twist_scale", config.design_space.twist_scale_range),
    ]

    for name, bounds in continuous_specs:
        x0 = float(getattr(base_design, name))
        step = 0.02 * float(bounds[1] - bounds[0])
        x_minus = max(float(bounds[0]), x0 - step)
        x_plus = min(float(bounds[1]), x0 + step)
        if abs(x_plus - x_minus) < 1e-12:
            continue

        e_minus = get_eval(replace(base_design, **{name: x_minus}))
        e_plus = get_eval(replace(base_design, **{name: x_plus}))
        delta = x_plus - x_minus

        dcp = (e_plus.performance.cp - e_minus.performance.cp) / delta
        droot = (e_plus.performance.root_moment_nm - e_minus.performance.root_moment_nm) / delta
        dsol = (e_plus.performance.solidity_mean - e_minus.performance.solidity_mean) / delta
        dist_minus = dist_from_eval(e_minus)
        dist_plus = dist_from_eval(e_plus)
        ddist = (dist_plus - dist_minus) / delta
        impact = max(abs(dist_minus - base_dist), abs(dist_plus - base_dist))

        rows.append(
            {
                "parameter": name,
                "type": "continuous",
                "baseline_value": x0,
                "step": step,
                "dcp_dparam": dcp,
                "droot_moment_dparam": droot,
                "dsolidity_dparam": dsol,
                "dtradeoff_distance_dparam": ddist,
                "impact_tradeoff": impact,
                "notes": "central finite difference around selected design",
            }
        )

    blade_options = sorted(set(int(x) for x in config.design_space.blades_options))
    base_blades = int(base_design.blades)
    idx = blade_options.index(base_blades)
    lower = blade_options[idx - 1] if idx - 1 >= 0 else None
    upper = blade_options[idx + 1] if idx + 1 < len(blade_options) else None
    if lower is not None or upper is not None:
        if lower is not None and upper is not None:
            e_low = get_eval(replace(base_design, blades=lower))
            e_up = get_eval(replace(base_design, blades=upper))
            delta = float(upper - lower)
            dcp = (e_up.performance.cp - e_low.performance.cp) / delta
            droot = (e_up.performance.root_moment_nm - e_low.performance.root_moment_nm) / delta
            dsol = (e_up.performance.solidity_mean - e_low.performance.solidity_mean) / delta
            dist_low = dist_from_eval(e_low)
            dist_up = dist_from_eval(e_up)
            ddist = (dist_up - dist_low) / delta
            impact = max(abs(dist_low - base_dist), abs(dist_up - base_dist))
            note = f"two-sided integer difference using B={lower} and B={upper}"
        else:
            neighbor = lower if lower is not None else upper
            e_nb = get_eval(replace(base_design, blades=int(neighbor)))
            delta = float(int(neighbor) - base_blades)
            dcp = (e_nb.performance.cp - base.performance.cp) / delta
            droot = (e_nb.performance.root_moment_nm - base.performance.root_moment_nm) / delta
            dsol = (e_nb.performance.solidity_mean - base.performance.solidity_mean) / delta
            dist_nb = dist_from_eval(e_nb)
            ddist = (dist_nb - base_dist) / delta
            impact = abs(dist_nb - base_dist)
            note = f"one-sided integer difference using B={neighbor}"
        rows.append(
            {
                "parameter": "blades",
                "type": "integer",
                "baseline_value": float(base_blades),
                "step": 1.0,
                "dcp_dparam": dcp,
                "droot_moment_dparam": droot,
                "dsolidity_dparam": dsol,
                "dtradeoff_distance_dparam": ddist,
                "impact_tradeoff": impact,
                "notes": note,
            }
        )

    base_airfoil_idx = int(base_design.airfoil_idx)
    base_airfoil = config.design_space.airfoils[base_airfoil_idx]
    alt_rows: list[dict[str, object]] = []
    for idx_alt, airfoil in enumerate(config.design_space.airfoils):
        if idx_alt == base_airfoil_idx:
            continue
        e_alt = get_eval(replace(base_design, airfoil_idx=idx_alt))
        dist_alt = dist_from_eval(e_alt)
        alt_rows.append(
            {
                "airfoil": airfoil,
                "delta_cp": e_alt.performance.cp - base.performance.cp,
                "delta_root_moment": e_alt.performance.root_moment_nm - base.performance.root_moment_nm,
                "delta_solidity": e_alt.performance.solidity_mean - base.performance.solidity_mean,
                "delta_tradeoff_distance": dist_alt - base_dist,
                "abs_tradeoff_shift": abs(dist_alt - base_dist),
            }
        )
    if alt_rows:
        alt_df = pd.DataFrame(alt_rows)
        best_alt = alt_df.loc[alt_df["delta_tradeoff_distance"].idxmin()]
        max_shift = float(alt_df["abs_tradeoff_shift"].max())
        rows.append(
            {
                "parameter": "airfoil",
                "type": "categorical",
                "baseline_value": base_airfoil,
                "step": float("nan"),
                "dcp_dparam": float("nan"),
                "droot_moment_dparam": float("nan"),
                "dsolidity_dparam": float("nan"),
                "dtradeoff_distance_dparam": float("nan"),
                "impact_tradeoff": max_shift,
                "notes": (
                    f"best alternate={best_alt['airfoil']} "
                    f"(delta_tradeoff_distance={best_alt['delta_tradeoff_distance']:.6f}), "
                    f"max_abs_tradeoff_shift={max_shift:.6f}"
                ),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty and "impact_tradeoff" in df.columns:
        df = df.sort_values("impact_tradeoff", ascending=False).reset_index(drop=True)
    return df


def run_pipeline(config: Config) -> dict[str, object]:
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
    sensitivity_csv = out_dir / "local_sensitivity.csv"

    all_df.to_csv(all_csv, index=False)
    pareto_df.to_csv(pareto_csv, index=False)
    sections_df.to_csv(section_csv, index=False)
    best = outcome.best_compromise
    airfoil_coords_df = polar_db.get_airfoil_coordinates(best.airfoil)
    airfoil_coords_df.to_csv(airfoil_coords_csv, index=False)
    _plot_sections(sections_df, section_png)
    _plot_pareto(pareto_df, pareto_png)
    _plot_airfoil_profile(airfoil_coords_df, airfoil_profile_png, title=f"Airfoil Profile: NACA {best.airfoil}")
    sensitivity_df = _compute_local_sensitivity(config=config, outcome=outcome, polar_db=polar_db)
    sensitivity_df.to_csv(sensitivity_csv, index=False)

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
            "local_sensitivity_csv": str(sensitivity_csv),
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
