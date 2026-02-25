from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from windturbine.config import DesignConfig
from windturbine.rotor.bem import RotorSection, design_rotor_sections, evaluate_rotor, find_alpha_opt
from windturbine.rotor.optimizer import OptimizationResult
from windturbine.rotor.polars import PolarModel


def _plot_twist_distribution(result: OptimizationResult, output_dir: Path) -> Path:
    out = output_dir / "twist_distribution.png"
    r_over_r = [sec.r_over_R for sec in result.section_results]
    twist = [sec.twist_deg for sec in result.section_results]

    plt.figure(figsize=(6, 4))
    plt.plot(r_over_r, twist, "-o", ms=3, lw=1.4, color="tab:blue")
    plt.xlabel("r/R")
    plt.ylabel("Twist (deg)")
    plt.title("Twist Distribution")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def _alpha_ld_table(polar: PolarModel) -> tuple[str, float]:
    alpha_opt = find_alpha_opt(polar)
    probe = sorted({2.0, 4.0, 6.0, 8.0, 10.0, round(alpha_opt, 2)})
    lines = ["| alpha (deg) | Cl | Cd | Cl/Cd |", "|---:|---:|---:|---:|"]
    for a_deg in probe:
        cl, cd = polar.coefficients(math.radians(float(a_deg)))
        ld = cl / cd if cd > 1e-12 else float("inf")
        lines.append(f"| {a_deg:.2f} | {cl:.3f} | {cd:.4f} | {ld:.2f} |")
    return "\n".join(lines), alpha_opt


def _cp_by_blade_count(config: DesignConfig, result: OptimizationResult, polar: PolarModel) -> str:
    lines = ["| B | Cp | Ct |", "|---:|---:|---:|"]
    for b in (2, 3, 4):
        sections = design_rotor_sections(
            radius_m=config.rotor.radius_m,
            hub_radius_ratio=result.hub_radius_ratio,
            n_sections=config.rotor.n_sections,
            blades=b,
            tip_speed_ratio=result.tip_speed_ratio,
            design_aoa_deg=result.aoa_deg,
            pitch_deg=config.rotor.pitch_deg,
            polar=polar,
        )
        clipped = [
            RotorSection(
                r_over_R=s.r_over_R,
                r_m=s.r_m,
                chord_m=min(s.chord_m, result.max_chord_m),
                twist_deg=s.twist_deg,
            )
            for s in sections
        ]
        _, perf = evaluate_rotor(
            sections=clipped,
            blades=b,
            radius_m=config.rotor.radius_m,
            hub_radius_ratio=result.hub_radius_ratio,
            tip_speed_ratio=result.tip_speed_ratio,
            wind_speed_ms=config.rotor.wind_speed_ms,
            pitch_deg=config.rotor.pitch_deg,
            air_density=config.rotor.air_density,
            dynamic_viscosity=config.rotor.dynamic_viscosity,
            polar=polar,
        )
        lines.append(f"| {b} | {perf.cp:.3f} | {perf.ct:.3f} |")
    return "\n".join(lines)


def build_report_markdown(config: DesignConfig, result: OptimizationResult) -> str:
    """Build report content and create supporting figures in outputs/.

    Includes:
    - Blade profile justification (Pritchard 11-parameter model)
    - AoA selection via L/D table
    - Twist formula + twist plot
    - Blade-number tradeoff + Cp(B) table
    - Methods section (surrogate + multi-objective optimization)
    - Assumptions and limitations
    """
    output_dir = (Path.cwd() / "outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    twist_plot_path = _plot_twist_distribution(result, output_dir)
    polar = PolarModel.from_config(config.polar)
    alpha_table, alpha_opt_from_polar = _alpha_ld_table(polar)
    blade_cp_table = _cp_by_blade_count(config, result, polar)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    a = config.airfoil
    conv = sum(1 for s in result.section_results if s.converged) / max(1, len(result.section_results))

    return f"""# Wind Turbine Design Report

Generated: {timestamp}

## Selected Design

- Blade count: **{result.blades}**
- Tip speed ratio: **{result.tip_speed_ratio:.2f}**
- Design angle of attack: **{result.aoa_deg:.2f} deg**
- Hub cutout: **r0/R = {result.hub_radius_ratio:.3f}**
- Max chord constraint: **{result.max_chord_m:.3f} m**
- Predicted power coefficient: **Cp = {result.performance.cp:.3f}**
- Predicted thrust coefficient: **Ct = {result.performance.ct:.3f}**
- Power at {config.rotor.wind_speed_ms:.1f} m/s: **{result.performance.power_w/1000.0:.1f} kW**
- Section convergence: **{100.0*conv:.1f}%**

## Blade Profile (Pritchard 11-Parameter Model)

Blade section geometry is generated with the 11-parameter Pritchard method
(*L.J. Pritchard, An eleven parameter axial turbine airfoil geometry model*),
using circle-and-cubic construction for pressure/suction surfaces.

Chosen parameters:
- radius = {a.radius}
- axial_chord = {a.axial_chord}
- tangential_chord = {a.tangential_chord}
- unguided_turning = {a.unguided_turning_deg} deg
- inlet_blade = {a.inlet_blade_deg} deg
- inlet_half_wedge = {a.inlet_half_wedge_deg} deg
- leading_edge_radius = {a.leading_edge_radius}
- outlet_blade = {a.outlet_blade_deg} deg
- trailing_edge_radius = {a.trailing_edge_radius}
- cascade_blade_count = {a.cascade_blade_count}
- throat = {a.throat}

![Generated airfoil](airfoil_plot.png)

## Angle of Attack Selection

The design AoA is selected by maximizing aerodynamic efficiency proxy **L/D = Cl/Cd**
within pre-stall limits from the 2D polar model.

Polar sweep summary:
{alpha_table}

Selected by polar max L/D: **alpha_opt = {alpha_opt_from_polar:.2f} deg**  
Selected in final optimized design: **{result.aoa_deg:.2f} deg**

## Twist Distribution

Twist follows BEM kinematics:

\\[
\\beta(r) = \\phi(r) - \\alpha_{{opt}} - \\theta_{{pitch}}
\\]

where \\(\\beta(r)\\) is local twist and \\(\\phi(r)\\) is inflow angle from the iterative BEM solve.

![Twist distribution]({twist_plot_path.name})

## Blade Number Trade-Off (B = 2, 3, 4)

Blade-count effect combines aerodynamic and structural trade-offs:
- Higher **B** raises solidity (more loading area).
- Higher **B** can reduce per-blade loading but increases material and drag.
- Tip/root loss behavior changes with **B** through Prandtl correction.
- Lower **B** may improve high-speed efficiency but worsen starting torque.

Cp comparison from sweep at the selected TSR/AoA:
{blade_cp_table}

## Methods (Paper-Like Architecture)

1. **Environment (fast evaluator):** BEM solver replaces CFD for rapid design evaluation.  
2. **Surrogate model:** MLP regressor maps design variables
   \\((B, \\lambda, \\alpha_{{opt}}, r_0/R, c_{{max}}/R)\\)
   to objectives \\((C_p, M_{{root}}, \\sigma, N_{{proxy}})\\).  
3. **Optimizer:** NSGA-II performs multi-objective search on the surrogate to approximate Pareto front.  
4. **Validation:** top Pareto candidates are re-evaluated with true BEM.  
5. **Dynamic re-optimization:** when constraints change, rerun NSGA-II on surrogate predictions for fast updates, then validate only shortlisted designs.

## Assumptions & Limitations

- BEM assumes annular, quasi-steady flow and momentum/blade-element coupling.
- 2D polar model is used instead of full 3D rotational/stall-delay corrections.
- Dynamic inflow, yawed flow, turbulence transients, and aeroelastic coupling are not modeled.
- Root moment and noise are proxies, not high-fidelity structural/acoustic simulations.
- Surrogate quality depends on training-domain coverage and may degrade out of sample.
"""
