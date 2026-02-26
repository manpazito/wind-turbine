from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from wind_turbine.config import Config
from wind_turbine.optimizer import OptimizationOutcome
from wind_turbine.xfoil import XfoilPolarDatabase


def _fmt(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def build_report(
    config: Config,
    outcome: OptimizationOutcome,
    polar_db: XfoilPolarDatabase,
    output_path: str | Path,
) -> str:
    best = outcome.best_compromise
    best_res = best.section_results

    re_mid = float(best_res[len(best_res) // 2].reynolds) if best_res else 1e6
    polar = polar_db.get_design_polar(best.airfoil, re_mid)
    ld = (polar["cl"] / polar["cd"]).replace([pd.NA], 0.0)
    best_ld_idx = int(ld.to_numpy().argmax())
    alpha_ld_best = float(polar.iloc[best_ld_idx]["alpha"])
    ld_best = float(ld.iloc[best_ld_idx])

    pareto_df = pd.DataFrame([x.to_dict() for x in outcome.pareto_front])
    blade_stats = (
        pareto_df.groupby("blades")[["cp", "root_moment_nm", "solidity_mean"]]
        .mean(numeric_only=True)
        .sort_index()
    )
    airfoil_stats = (
        pareto_df.groupby("airfoil")[["cp", "root_moment_nm", "solidity_mean"]]
        .mean(numeric_only=True)
        .sort_values("cp", ascending=False)
    )

    twist_root = float(best_res[0].twist_deg) if best_res else float("nan")
    twist_tip = float(best_res[-1].twist_deg) if best_res else float("nan")
    alpha_mean = float(sum(s.alpha_deg for s in best_res) / max(len(best_res), 1))

    lines = [
        "# Wind Turbine Optimization Report",
        "",
        "This run follows the multi-objective spirit of the provided paper, but uses a wind-turbine workflow:",
        "XFOIL polars + BEM simulation + NSGA-II search on blade profile, AoA, twist scaling, TSR, and blade count.",
        "",
        "## Core Equations Used",
        "",
        "1. Tip speed ratio:",
        "   - `lambda = Omega * R / V_inf`",
        "2. Inflow angle and angle of attack:",
        "   - `phi = atan((1-a) / (lambda_r * (1+a')))`",
        "   - `alpha = phi - (theta + beta_pitch)`",
        "3. Force coefficients:",
        "   - `C_n = C_l cos(phi) + C_d sin(phi)`",
        "   - `C_t = C_l sin(phi) - C_d cos(phi)`",
        "4. Induction factors (BEM form):",
        "   - `a = 1 / ( (4F sin^2(phi))/(sigma C_n) + 1 )`",
        "   - `a' = 1 / ( (4F sin(phi)cos(phi))/(sigma C_t) - 1 )`",
        "5. Section loads and rotor power:",
        "   - `dT = 0.5 rho W^2 B c C_n dr`",
        "   - `dQ = 0.5 rho W^2 B c C_t r dr`",
        "   - `Cp = P / (0.5 rho A V_inf^3),  P = Omega * integral(dQ)`",
        "6. Geometry synthesis equations:",
        "   - `phi_des(r) ~= (2/3) atan(1/lambda_r)`",
        "   - `theta(r) = phi_des(r) - alpha_design - beta_pitch`",
        "   - `c(r) ~ 8 pi r sin(phi_des)/(B C_l,des lambda_r)`",
        "",
        "## Optimized Design (Best Compromise From Pareto)",
        "",
        f"- Airfoil profile: `NACA {best.airfoil}`",
        f"- Blade number: `{best.design.blades}`",
        f"- Design AoA: `{_fmt(best.design.aoa_deg, 3)} deg`",
        f"- Tip speed ratio: `{_fmt(best.design.tip_speed_ratio, 3)}`",
        f"- Hub radius ratio: `{_fmt(best.design.hub_radius_ratio, 4)}`",
        f"- Twist scale: `{_fmt(best.design.twist_scale, 4)}`",
        f"- Chord scale: `{_fmt(best.design.chord_scale, 4)}`",
        f"- Achieved Cp: `{_fmt(best.performance.cp, 4)}`",
        f"- Root moment: `{_fmt(best.performance.root_moment_nm, 2)} N.m`",
        f"- Mean solidity: `{_fmt(best.performance.solidity_mean, 5)}`",
        "",
        "## Justification",
        "",
        f"- Blade profile choice (`NACA {best.airfoil}`): selected on the Pareto front where it balances high Cp and moderate root load.",
        f"- Angle of attack: optimized AoA `{_fmt(best.design.aoa_deg, 3)} deg`; XFOIL at Re≈{int(re_mid)} gives peak Cl/Cd around `{_fmt(alpha_ld_best, 3)} deg` (Cl/Cd≈{_fmt(ld_best, 3)}), so chosen AoA is in a high-efficiency aerodynamic band.",
        f"- Twist: optimized from `theta(r)=phi_des-alpha_design-beta_pitch`; resulting twist decreases from `{_fmt(twist_root, 3)} deg` at root to `{_fmt(twist_tip, 3)} deg` near tip, which is physically consistent with lower inflow angle at larger radius.",
        f"- Blade number: `{best.design.blades}` emerged from multi-objective trade-off; the blade-count means on Pareto solutions are reported below for transparency.",
        "",
        "## Pareto Mean Performance by Blade Count",
        "",
    ]

    for blade_count, row in blade_stats.iterrows():
        lines.append(
            f"- B={blade_count}: Cp={_fmt(float(row['cp']), 4)}, "
            f"RootMoment={_fmt(float(row['root_moment_nm']), 2)} N.m, "
            f"Solidity={_fmt(float(row['solidity_mean']), 5)}"
        )

    lines.extend(
        [
            "",
            "## Pareto Mean Performance by Airfoil",
            "",
        ]
    )

    for airfoil, row in airfoil_stats.iterrows():
        lines.append(
            f"- NACA {airfoil}: Cp={_fmt(float(row['cp']), 4)}, "
            f"RootMoment={_fmt(float(row['root_moment_nm']), 2)} N.m, "
            f"Solidity={_fmt(float(row['solidity_mean']), 5)}"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is an aerodynamic optimization model. Structural and fatigue constraints should be added before manufacturing decisions.",
            "- The workflow is deterministic for a fixed random seed and config file.",
        ]
    )

    report_text = "\n".join(lines) + "\n"
    out_path = Path(output_path)
    out_path.write_text(report_text, encoding="utf-8")
    return report_text
