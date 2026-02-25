from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PolarModel:
    """Analytic 2D polar model for conceptual BEM design.

    References:
    - Thin-airfoil lift slope, Cl ~= 2*pi*alpha (Anderson, Fundamentals of Aerodynamics).
    - Parabolic drag polar, Cd = Cd0 + k*Cl^2 (standard preliminary airfoil modeling,
      e.g., Abbott and von Doenhoff).
    """

    cl_alpha_per_rad: float = 5.73
    alpha_stall_deg: float = 12.0
    cl_max: float = 1.35
    cd0: float = 0.01
    induced_drag_factor: float = 0.012
    post_stall_drag_gain: float = 0.02

    @classmethod
    def from_config(cls, cfg: object) -> "PolarModel":
        return cls(
            cl_alpha_per_rad=float(getattr(cfg, "cl_alpha_per_rad")),
            alpha_stall_deg=float(getattr(cfg, "alpha_stall_deg")),
            cl_max=float(getattr(cfg, "cl_max")),
            cd0=float(getattr(cfg, "cd0")),
            induced_drag_factor=float(getattr(cfg, "induced_drag_factor")),
            post_stall_drag_gain=float(getattr(cfg, "post_stall_drag_gain")),
        )

    def coefficients(self, alpha_rad: float, reynolds: float | None = None) -> tuple[float, float]:
        alpha_deg = math.degrees(alpha_rad)
        alpha_abs = abs(alpha_deg)

        cl_linear = self.cl_alpha_per_rad * alpha_rad
        if alpha_abs <= self.alpha_stall_deg:
            cl = cl_linear
        else:
            sign = 1.0 if cl_linear >= 0.0 else -1.0
            excess = (alpha_abs - self.alpha_stall_deg) / max(self.alpha_stall_deg, 1e-6)
            cl = sign * self.cl_max * (1.0 - 0.15 * min(excess, 1.0))

        cl = max(-self.cl_max, min(self.cl_max, cl))
        cd = self.cd0 + self.induced_drag_factor * cl * cl

        if alpha_abs > self.alpha_stall_deg:
            stall_ratio = alpha_abs / max(self.alpha_stall_deg, 1e-6)
            cd += self.post_stall_drag_gain * (stall_ratio**2)

        if reynolds is not None and reynolds > 0.0:
            cd *= max(0.85, min(1.15, 1.0 + 2e5 / reynolds * 0.02))

        return cl, cd
