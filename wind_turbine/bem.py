from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wind_turbine.config import DesignSpaceConfig, RotorConfig
from wind_turbine.xfoil import PolarPoint, XfoilPolarDatabase


@dataclass(frozen=True)
class SectionGeometry:
    r_m: float
    r_over_r: float
    chord_m: float
    twist_deg: float


@dataclass(frozen=True)
class SectionResult:
    r_m: float
    chord_m: float
    twist_deg: float
    phi_deg: float
    alpha_deg: float
    reynolds: float
    cl: float
    cd: float
    a: float
    a_prime: float
    cn: float
    ct: float
    dthrust_n: float
    dtorque_nm: float
    local_solidity: float


@dataclass(frozen=True)
class RotorPerformance:
    cp: float
    ct: float
    power_w: float
    thrust_n: float
    torque_nm: float
    root_moment_nm: float
    solidity_mean: float


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(min(max(value, lo), hi))


def _prandtl_loss_factor(
    blades: int,
    radius_m: float,
    hub_radius_m: float,
    r_m: float,
    phi_rad: float,
) -> float:
    sin_phi = abs(math.sin(phi_rad))
    sin_phi = max(sin_phi, 1e-6)
    denom = max(r_m * sin_phi, 1e-6)
    tip_exp = -0.5 * blades * (radius_m - r_m) / denom
    root_exp = -0.5 * blades * (r_m - hub_radius_m) / denom
    tip_term = math.exp(_clamp(tip_exp, -60.0, 0.0))
    root_term = math.exp(_clamp(root_exp, -60.0, 0.0))
    tip_loss = (2.0 / math.pi) * math.acos(_clamp(tip_term, 0.0, 1.0))
    root_loss = (2.0 / math.pi) * math.acos(_clamp(root_term, 0.0, 1.0))
    return max(tip_loss * root_loss, 1e-3)


def design_blade_geometry(
    rotor: RotorConfig,
    design_space: DesignSpaceConfig,
    polar_db: XfoilPolarDatabase,
    airfoil: str,
    blades: int,
    tip_speed_ratio: float,
    design_aoa_deg: float,
    hub_radius_ratio: float,
    chord_scale: float,
    twist_scale: float,
) -> list[SectionGeometry]:
    """
    Build an initial geometry using common Glauert/Schmitz design relations:
      phi(r) ~= (2/3) * arctan(1/lambda_r)
      theta(r) = phi(r) - alpha_design - pitch
      c(r) ~ 8*pi*r*sin(phi) / (B*Cl_design*lambda_r)
    """
    radius_m = rotor.radius_m
    hub_radius_m = hub_radius_ratio * radius_m
    dr = (radius_m - hub_radius_m) / rotor.n_sections
    min_chord = design_space.chord_ratio_limits[0] * radius_m
    max_chord = design_space.chord_ratio_limits[1] * radius_m

    sections: list[SectionGeometry] = []
    for i in range(rotor.n_sections):
        r_m = hub_radius_m + (i + 0.5) * dr
        lambda_r = tip_speed_ratio * (r_m / radius_m)
        phi_des = (2.0 / 3.0) * math.atan2(1.0, max(lambda_r, 1e-6))

        re_guess = 3.5e5 + 2.8e6 * (r_m / radius_m)
        cl_design = max(polar_db.sample(airfoil, re_guess, design_aoa_deg).cl, 0.35)

        chord_m = chord_scale * (
            8.0 * math.pi * r_m * math.sin(phi_des) / (blades * cl_design * max(lambda_r, 1e-6))
        )
        chord_m = _clamp(chord_m, min_chord, max_chord)

        twist_base = math.degrees(phi_des) - design_aoa_deg - rotor.pitch_deg
        twist_deg = twist_base * twist_scale
        sections.append(
            SectionGeometry(
                r_m=r_m,
                r_over_r=r_m / radius_m,
                chord_m=chord_m,
                twist_deg=twist_deg,
            )
        )
    return sections


def evaluate_rotor(
    rotor: RotorConfig,
    polar_db: XfoilPolarDatabase,
    airfoil: str,
    blades: int,
    tip_speed_ratio: float,
    hub_radius_ratio: float,
    sections: list[SectionGeometry],
) -> tuple[list[SectionResult], RotorPerformance]:
    radius_m = rotor.radius_m
    hub_radius_m = hub_radius_ratio * radius_m
    omega = tip_speed_ratio * rotor.wind_speed_ms / radius_m
    dr = (radius_m - hub_radius_m) / max(len(sections), 1)

    thrust_n = 0.0
    torque_nm = 0.0
    root_moment_nm = 0.0
    section_results: list[SectionResult] = []
    solidity_terms: list[float] = []

    for sec in sections:
        a = 0.30
        a_prime = 0.00
        last: SectionResult | None = None

        for _ in range(120):
            v_axial = rotor.wind_speed_ms * (1.0 - a)
            v_tan = omega * sec.r_m * (1.0 + a_prime)
            phi = math.atan2(max(v_axial, 1e-8), max(v_tan, 1e-8))
            w_rel = math.hypot(v_axial, v_tan)
            alpha_deg = math.degrees(phi) - (sec.twist_deg + rotor.pitch_deg)
            reynolds = rotor.air_density * w_rel * sec.chord_m / rotor.dynamic_viscosity
            polar: PolarPoint = polar_db.sample(airfoil, reynolds, alpha_deg)
            cl = polar.cl
            cd = polar.cd

            cn = cl * math.cos(phi) + cd * math.sin(phi)
            ct = cl * math.sin(phi) - cd * math.cos(phi)
            sigma = blades * sec.chord_m / (2.0 * math.pi * sec.r_m)
            f_loss = _prandtl_loss_factor(blades, radius_m, hub_radius_m, sec.r_m, phi)

            sin_phi = max(abs(math.sin(phi)), 1e-5)
            cos_phi = max(abs(math.cos(phi)), 1e-5)
            denom_a = (4.0 * f_loss * sin_phi * sin_phi) / max(sigma * cn, 1e-8)
            a_new = 1.0 / (denom_a + 1.0)
            denom_ap = (4.0 * f_loss * sin_phi * cos_phi) / max(sigma * ct, 1e-8)
            a_prime_new = 1.0 / (denom_ap - 1.0)

            a_new = _clamp(a_new, 0.0, 0.95)
            a_prime_new = _clamp(a_prime_new, -0.5, 0.5)

            a_upd = 0.75 * a + 0.25 * a_new
            ap_upd = 0.75 * a_prime + 0.25 * a_prime_new

            dyn_pressure = 0.5 * rotor.air_density * (w_rel**2)
            d_lift = dyn_pressure * sec.chord_m * cl * dr
            d_drag = dyn_pressure * sec.chord_m * cd * dr
            dthrust_n = blades * (d_lift * math.cos(phi) + d_drag * math.sin(phi))
            dtorque_nm = blades * (d_lift * math.sin(phi) - d_drag * math.cos(phi)) * sec.r_m

            last = SectionResult(
                r_m=sec.r_m,
                chord_m=sec.chord_m,
                twist_deg=sec.twist_deg,
                phi_deg=math.degrees(phi),
                alpha_deg=alpha_deg,
                reynolds=float(reynolds),
                cl=float(cl),
                cd=float(cd),
                a=float(a_upd),
                a_prime=float(ap_upd),
                cn=float(cn),
                ct=float(ct),
                dthrust_n=float(dthrust_n),
                dtorque_nm=float(dtorque_nm),
                local_solidity=float(sigma),
            )

            if abs(a_upd - a) < 1e-4 and abs(ap_upd - a_prime) < 1e-4:
                a = a_upd
                a_prime = ap_upd
                break
            a = a_upd
            a_prime = ap_upd

        if last is None:
            continue

        thrust_n += last.dthrust_n
        torque_nm += last.dtorque_nm
        root_moment_nm += last.dthrust_n * max(sec.r_m - hub_radius_m, 0.0)
        solidity_terms.append(last.local_solidity)
        section_results.append(last)

    swept_area = math.pi * (radius_m**2)
    power_w = omega * torque_nm
    cp = power_w / (0.5 * rotor.air_density * swept_area * rotor.wind_speed_ms**3 + 1e-9)
    ct = thrust_n / (0.5 * rotor.air_density * swept_area * rotor.wind_speed_ms**2 + 1e-9)
    solidity_mean = float(np.mean(solidity_terms)) if solidity_terms else 0.0

    perf = RotorPerformance(
        cp=float(cp),
        ct=float(ct),
        power_w=float(power_w),
        thrust_n=float(thrust_n),
        torque_nm=float(torque_nm),
        root_moment_nm=float(root_moment_nm),
        solidity_mean=solidity_mean,
    )
    return section_results, perf
