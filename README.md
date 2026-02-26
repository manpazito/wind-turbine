# Wind Turbine Blade Optimization (From Zero)

This project builds a full wind-turbine aerodynamic optimization model from scratch using:

- `XFOIL` for airfoil polars (`Cl`, `Cd`)
- `BEM` (Blade Element Momentum) for rotor performance
- `NSGA-II` style multi-objective optimization

It optimizes and justifies:

- blade profile (airfoil)
- angle of attack
- twist distribution
- blade number

under equations commonly used in wind-turbine research.

## 1) Install

```bash
python -m pip install -r requirements.txt
```

Make sure `xfoil` is installed and available on your `PATH`.

## 2) Run

```bash
python -m wind_turbine --config configs/default.yaml
```

Preset run profiles:

```bash
# Super fast smoke test
python -m wind_turbine --config configs/quick_test.yaml

# High-accuracy (slow, much denser optimization + XFOIL sampling)
python -m wind_turbine --config configs/high_accuracy.yaml
```

## 3) Outputs

Generated in `outputs/`:

- `all_designs.csv`: all evaluated designs
- `pareto.csv`: Pareto-optimal designs
- `best_sections.csv`: radial geometry and section aerodynamic states
- `best_airfoil_coords.csv`: x,y coordinates of selected blade profile (normalized by chord)
- `best_geometry.png`: chord and twist distributions
- `best_airfoil_profile.png`: 2D airfoil/profile plot from x,y coordinates
- `pareto_cp_vs_moment.png`: Pareto scatter
- `summary.json`: selected best compromise design
- `report.md`: equation-based justification for profile, AoA, twist, and blade count

## 4) Equations used (implemented)

- `lambda = Omega * R / V_inf`
- `phi = atan((1-a) / (lambda_r * (1+a')))`
- `alpha = phi - (theta + beta_pitch)`
- `C_n = C_l cos(phi) + C_d sin(phi)`
- `C_t = C_l sin(phi) - C_d cos(phi)`
- `a = 1 / ((4 F sin^2(phi))/(sigma C_n) + 1)`
- `a' = 1 / ((4 F sin(phi)cos(phi))/(sigma C_t) - 1)`
- `dT = 0.5 rho W^2 B c C_n dr`
- `dQ = 0.5 rho W^2 B c C_t r dr`
- `Cp = P / (0.5 rho A V_inf^3), P = Omega * integral(dQ)`
- `theta(r) = phi_des(r) - alpha_design - beta_pitch`
- `c(r) ~ 8 pi r sin(phi_des)/(B C_l,des lambda_r)`

## 5) Notes

- The model is aerodynamic-first. Structural, fatigue, and manufacturing constraints should be added for final engineering design.
- Configurable ranges are in `configs/default.yaml`.
