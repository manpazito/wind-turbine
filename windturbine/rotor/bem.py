from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd

from windturbine.rotor.polars import PolarModel


class PolarLike(Protocol):
    def coefficients(self, alpha_rad: float, reynolds: float | None = None) -> tuple[float, float]:
        ...


@dataclass(frozen=True)
class RotorSection:
    r_over_R: float
    r_m: float
    chord_m: float
    twist_deg: float


@dataclass
class SectionResult:
    r_over_R: float
    r_m: float
    chord_m: float
    twist_deg: float
    aoa_deg: float
    phi_deg: float
    cl: float
    cd: float
    reynolds: float
    local_solidity: float
    converged: bool
    iterations: int
    dthrust_n: float
    dtorque_nm: float


@dataclass
class RotorPerformance:
    cp: float
    ct: float
    power_w: float
    thrust_n: float
    torque_nm: float


def _polar_coefficients(polar: Any, alpha_rad: float, reynolds: float | None = None) -> tuple[float, float]:
    if hasattr(polar, "coefficients"):
        return polar.coefficients(alpha_rad, reynolds)
    if callable(polar):
        return polar(alpha_rad, reynolds)
    raise TypeError("polar must provide .coefficients(alpha_rad, reynolds) or be callable")


def prandtl_loss_factor(
    blades: int,
    r_m: float,
    radius_m: float,
    hub_radius_m: float,
    phi_rad: float,
) -> float:
    """Combined tip/root Prandtl correction.

    Reference: Glauert and standard BEM texts (Hansen; Burton et al.).
    """
    sin_phi = max(abs(math.sin(phi_rad)), 1e-7)
    r_m = max(r_m, hub_radius_m + 1e-7)
    hub_radius_m = max(hub_radius_m, 1e-7)

    tip_arg = (blades / 2.0) * ((radius_m - r_m) / (r_m * sin_phi))
    root_arg = (blades / 2.0) * ((r_m - hub_radius_m) / (hub_radius_m * sin_phi))

    f_tip = (2.0 / math.pi) * math.acos(min(1.0, max(0.0, math.exp(-max(tip_arg, 0.0)))))
    f_root = (2.0 / math.pi) * math.acos(min(1.0, max(0.0, math.exp(-max(root_arg, 0.0)))))
    return max(1e-3, f_tip * f_root)


def find_alpha_opt(polar: Any, stall_limit_deg: float | None = None) -> float:
    """Choose alpha maximizing Cl/Cd inside pre-stall range.

    Uses a dense sweep and returns the optimal angle in degrees.
    """
    if stall_limit_deg is None:
        stall_limit_deg = float(getattr(polar, "alpha_stall_deg", 12.0))

    alphas_deg = np.linspace(-2.0, stall_limit_deg, 220)
    best_alpha = 6.0
    best_ld = -np.inf

    for alpha_deg in alphas_deg:
        cl, cd = _polar_coefficients(polar, math.radians(float(alpha_deg)), None)
        if cd <= 0.0:
            continue
        ld = cl / cd
        if ld > best_ld:
            best_ld = ld
            best_alpha = float(alpha_deg)

    return best_alpha


def _omega_from_inputs(radius_m: float, wind_speed_ms: float, tip_speed_ratio: float | None, omega: float | None) -> tuple[float, float]:
    if omega is not None:
        omega_val = float(omega)
        tsr_val = omega_val * radius_m / max(wind_speed_ms, 1e-9)
        return omega_val, tsr_val
    if tip_speed_ratio is None:
        raise ValueError("Provide either omega or tip_speed_ratio/TSR")
    tsr_val = float(tip_speed_ratio)
    omega_val = tsr_val * wind_speed_ms / max(radius_m, 1e-9)
    return omega_val, tsr_val


def _build_radial_stations(radius_m: float, hub_radius_ratio: float, n_sections: int) -> np.ndarray:
    r_root = hub_radius_ratio * radius_m
    dr = (radius_m - r_root) / max(n_sections, 1)
    return r_root + (np.arange(n_sections) + 0.5) * dr


def design_rotor_sections(
    radius_m: float,
    hub_radius_ratio: float,
    n_sections: int,
    blades: int,
    tip_speed_ratio: float,
    design_aoa_deg: float,
    pitch_deg: float,
    polar: PolarLike,
    target_axial_induction: float = 1.0 / 3.0,
) -> list[RotorSection]:
    """Glauert-style optimal design from alpha_opt and TSR.

    Uses a common optimum relationship for preliminary HAWT design:
      phi = atan((1-a)/(lambda_r*(1+a'))), with a ~= 1/3, a' ~= 0
      twist(r) = phi - alpha_opt - pitch
      chord(r) = (8*pi*r*F*sin(phi)*cos(phi)) / (B*Cl*lambda_r)

    References:
    - M. O. L. Hansen, Aerodynamics of Wind Turbines.
    - T. Burton et al., Wind Energy Handbook.
    """
    r = _build_radial_stations(radius_m, hub_radius_ratio, n_sections)
    hub_radius_m = hub_radius_ratio * radius_m
    a0 = target_axial_induction
    a0p = 0.0

    alpha_opt_rad = math.radians(design_aoa_deg)
    cl_opt, _ = _polar_coefficients(polar, alpha_opt_rad, None)
    cl_opt = max(abs(cl_opt), 0.15)

    sections: list[RotorSection] = []
    for ri in r:
        r_over_R = float(ri / radius_m)
        lambda_r = max(1e-6, tip_speed_ratio * r_over_R)
        phi = math.atan2((1.0 - a0), lambda_r * (1.0 + a0p))
        F = prandtl_loss_factor(blades, float(ri), radius_m, hub_radius_m, phi)

        # Glauert-style design chord with Prandtl factor.
        chord = (8.0 * math.pi * ri * F * (math.sin(phi) ** 2)) / (
            max(blades * cl_opt * lambda_r, 1e-8)
        )
        chord = float(np.clip(chord, 0.0001 * radius_m, 0.12 * radius_m))

        twist_deg = math.degrees(phi) - design_aoa_deg - pitch_deg
        sections.append(
            RotorSection(
                r_over_R=r_over_R,
                r_m=float(ri),
                chord_m=chord,
                twist_deg=float(twist_deg),
            )
        )

    return sections


def solve_section(
    section: RotorSection,
    blades: int,
    radius_m: float,
    hub_radius_m: float,
    omega_rad_s: float,
    wind_speed_ms: float,
    pitch_deg: float,
    air_density: float,
    dynamic_viscosity: float,
    dr: float,
    polar: Any,
    max_iter: int = 150,
    tol: float = 1e-5,
    relaxation: float = 0.35,
) -> SectionResult:
    """Iterative BEM section solve for a, a'.

    Core relations:
    - phi = atan(((1-a)V_inf)/((1+a')*omega*r))
    - alpha = phi - (twist + pitch)
    - Cn = Cl cos(phi) + Cd sin(phi)
    - Ct = Cl sin(phi) - Cd cos(phi)
    - a  = 1 / (1 + 4F sin^2(phi)/(sigma Cn))
    - a' = 1 / (4F sin(phi)cos(phi)/(sigma Ct) - 1)

    References: Hansen; Burton; Glauert BEM derivation.
    """
    a = 0.30
    a_prime = 0.01
    converged = False
    iterations = 0

    for iterations in range(1, max_iter + 1):
        v_axial = wind_speed_ms * (1.0 - a)
        v_tan = omega_rad_s * section.r_m * (1.0 + a_prime)
        phi = math.atan2(max(v_axial, 1e-8), max(v_tan, 1e-8))
        alpha = phi - math.radians(section.twist_deg + pitch_deg)

        v_rel = math.hypot(v_axial, v_tan)
        reynolds = air_density * v_rel * section.chord_m / max(dynamic_viscosity, 1e-12)
        cl, cd = _polar_coefficients(polar, alpha, reynolds)

        c_n = cl * math.cos(phi) + cd * math.sin(phi)
        c_t = cl * math.sin(phi) - cd * math.cos(phi)
        sigma = blades * section.chord_m / (2.0 * math.pi * section.r_m)
        F = prandtl_loss_factor(blades, section.r_m, radius_m, hub_radius_m, phi)

        a_new = 1.0 / (1.0 + (4.0 * F * (math.sin(phi) ** 2)) / max(sigma * c_n, 1e-9))
        a_new = float(np.clip(a_new, 0.0, 0.95))

        denom_ap = (4.0 * F * math.sin(phi) * math.cos(phi)) / max(sigma * c_t, 1e-9) - 1.0
        if abs(denom_ap) < 1e-9:
            a_prime_new = a_prime
        else:
            a_prime_new = 1.0 / denom_ap
        a_prime_new = float(np.clip(a_prime_new, -0.5, 0.8))

        da = a_new - a
        dap = a_prime_new - a_prime
        a += relaxation * da
        a_prime += relaxation * dap

        if abs(da) < tol and abs(dap) < tol:
            converged = True
            break

    v_axial = wind_speed_ms * (1.0 - a)
    v_tan = omega_rad_s * section.r_m * (1.0 + a_prime)
    phi = math.atan2(max(v_axial, 1e-8), max(v_tan, 1e-8))
    alpha = phi - math.radians(section.twist_deg + pitch_deg)

    v_rel = math.hypot(v_axial, v_tan)
    reynolds = air_density * v_rel * section.chord_m / max(dynamic_viscosity, 1e-12)
    cl, cd = _polar_coefficients(polar, alpha, reynolds)

    c_n = cl * math.cos(phi) + cd * math.sin(phi)
    c_t = cl * math.sin(phi) - cd * math.cos(phi)
    sigma = blades * section.chord_m / (2.0 * math.pi * section.r_m)

    q_rel = 0.5 * air_density * v_rel * v_rel
    d_lift_normal = q_rel * section.chord_m * c_n
    d_lift_tan = q_rel * section.chord_m * c_t

    dthrust = blades * d_lift_normal * dr
    dtorque = blades * d_lift_tan * section.r_m * dr

    return SectionResult(
        r_over_R=section.r_over_R,
        r_m=section.r_m,
        chord_m=section.chord_m,
        twist_deg=section.twist_deg,
        aoa_deg=math.degrees(alpha),
        phi_deg=math.degrees(phi),
        cl=float(cl),
        cd=float(cd),
        reynolds=float(reynolds),
        local_solidity=float(sigma),
        converged=converged,
        iterations=iterations,
        dthrust_n=float(dthrust),
        dtorque_nm=float(dtorque),
    )


def evaluate_rotor(
    sections: list[RotorSection],
    blades: int,
    radius_m: float,
    hub_radius_ratio: float,
    tip_speed_ratio: float,
    wind_speed_ms: float,
    pitch_deg: float,
    air_density: float,
    dynamic_viscosity: float,
    polar: Any,
) -> tuple[list[SectionResult], RotorPerformance]:
    hub_radius_m = hub_radius_ratio * radius_m
    dr = (radius_m - hub_radius_m) / max(len(sections) - 1, 1)
    omega = tip_speed_ratio * wind_speed_ms / max(radius_m, 1e-9)

    section_results: list[SectionResult] = []
    for section in sections:
        section_results.append(
            solve_section(
                section=section,
                blades=blades,
                radius_m=radius_m,
                hub_radius_m=hub_radius_m,
                omega_rad_s=omega,
                wind_speed_ms=wind_speed_ms,
                pitch_deg=pitch_deg,
                air_density=air_density,
                dynamic_viscosity=dynamic_viscosity,
                dr=dr,
                polar=polar,
            )
        )

    thrust_n = float(sum(item.dthrust_n for item in section_results))
    torque_nm = float(sum(item.dtorque_nm for item in section_results))
    power_w = torque_nm * omega

    swept_area = math.pi * radius_m * radius_m
    q_inf = 0.5 * air_density * wind_speed_ms * wind_speed_ms
    cp = power_w / max(q_inf * swept_area * wind_speed_ms, 1e-9)
    ct = thrust_n / max(q_inf * swept_area, 1e-9)

    return section_results, RotorPerformance(cp=cp, ct=ct, power_w=power_w, thrust_n=thrust_n, torque_nm=torque_nm)


def _sections_to_dataframe(section_results: list[SectionResult]) -> pd.DataFrame:
    rows = []
    for sec in section_results:
        rows.append(
            {
                "r_over_R": sec.r_over_R,
                "r_m": sec.r_m,
                "chord_m": sec.chord_m,
                "twist_deg": sec.twist_deg,
                "aoa_deg": sec.aoa_deg,
                "phi_deg": sec.phi_deg,
                "cl": sec.cl,
                "cd": sec.cd,
                "reynolds": sec.reynolds,
                "local_solidity": sec.local_solidity,
            }
        )
    return pd.DataFrame(rows)


def write_rotor_sections_csv(section_results: list[SectionResult], path: str | Path = "outputs/rotor_sections.csv") -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = _sections_to_dataframe(section_results)
    df.to_csv(out_path, index=False)
    return out_path


def design_and_evaluate(
    radius_m: float,
    hub_radius_ratio: float,
    n_sections: int,
    blades: int,
    tip_speed_ratio: float,
    design_aoa_deg: float,
    wind_speed_ms: float,
    pitch_deg: float,
    air_density: float,
    dynamic_viscosity: float,
    polar: Any,
) -> tuple[list[RotorSection], list[SectionResult], RotorPerformance]:
    sections = design_rotor_sections(
        radius_m=radius_m,
        hub_radius_ratio=hub_radius_ratio,
        n_sections=n_sections,
        blades=blades,
        tip_speed_ratio=tip_speed_ratio,
        design_aoa_deg=design_aoa_deg,
        pitch_deg=pitch_deg,
        polar=polar,
    )
    section_results, performance = evaluate_rotor(
        sections=sections,
        blades=blades,
        radius_m=radius_m,
        hub_radius_ratio=hub_radius_ratio,
        tip_speed_ratio=tip_speed_ratio,
        wind_speed_ms=wind_speed_ms,
        pitch_deg=pitch_deg,
        air_density=air_density,
        dynamic_viscosity=dynamic_viscosity,
        polar=polar,
    )
    return sections, section_results, performance


def design_rotor(config: Mapping[str, Any] | Any) -> tuple[pd.DataFrame, float]:
    """High-level design mode API requested by user.

    Inputs accepted via `config`:
    - R, B, rho, V_inf
    - omega or TSR/tip_speed_ratio
    - pitch_deg
    - optional: alpha_opt_deg, hub_radius_ratio, n_sections, mu, polar, output_csv

    Returns `(section_dataframe, cp)` and writes `rotor_sections.csv`.
    """
    def getv(name: str, default: Any = None) -> Any:
        if isinstance(config, Mapping):
            return config.get(name, default)
        return getattr(config, name, default)

    radius_m = float(getv("R", getv("radius_m")))
    blades = int(getv("B", getv("blades", 3)))
    rho = float(getv("rho", getv("air_density", 1.225)))
    v_inf = float(getv("V_inf", getv("wind_speed_ms", 8.0)))
    pitch_deg = float(getv("pitch_deg", 0.0))
    hub_radius_ratio = float(getv("hub_radius_ratio", 0.2))
    n_sections = int(getv("n_sections", 20))
    mu = float(getv("mu", getv("dynamic_viscosity", 1.81e-5)))

    tip_speed_ratio = getv("TSR", getv("tip_speed_ratio", None))
    omega = getv("omega", getv("omega_rad_s", None))
    omega_rad_s, tsr = _omega_from_inputs(radius_m, v_inf, tip_speed_ratio, omega)

    polar_obj = getv("polar", PolarModel())
    alpha_opt_deg = getv("alpha_opt_deg", None)
    if alpha_opt_deg is None:
        alpha_opt_deg = find_alpha_opt(polar_obj)

    sections = design_rotor_sections(
        radius_m=radius_m,
        hub_radius_ratio=hub_radius_ratio,
        n_sections=n_sections,
        blades=blades,
        tip_speed_ratio=float(tsr),
        design_aoa_deg=float(alpha_opt_deg),
        pitch_deg=pitch_deg,
        polar=polar_obj,
    )

    section_results: list[SectionResult] = []
    dr = (radius_m - hub_radius_ratio * radius_m) / max(n_sections - 1, 1)
    hub_radius_m = hub_radius_ratio * radius_m
    for section in sections:
        section_results.append(
            solve_section(
                section=section,
                blades=blades,
                radius_m=radius_m,
                hub_radius_m=hub_radius_m,
                omega_rad_s=omega_rad_s,
                wind_speed_ms=v_inf,
                pitch_deg=pitch_deg,
                air_density=rho,
                dynamic_viscosity=mu,
                dr=dr,
                polar=polar_obj,
            )
        )

    thrust_n = float(sum(item.dthrust_n for item in section_results))
    torque_nm = float(sum(item.dtorque_nm for item in section_results))
    power_w = torque_nm * omega_rad_s

    area = math.pi * radius_m * radius_m
    cp = power_w / max(0.5 * rho * area * (v_inf**3), 1e-9)

    output_csv = Path(getv("output_csv", "outputs/rotor_sections.csv"))
    write_rotor_sections_csv(section_results, output_csv)

    # Also write a compact power summary for convenience.
    summary_path = output_csv.parent / "rotor_power_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cp", "power_w", "thrust_n", "torque_nm"])
        writer.writerow([cp, power_w, thrust_n, torque_nm])

    return _sections_to_dataframe(section_results), float(cp)
