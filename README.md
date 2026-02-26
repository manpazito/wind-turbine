# Wind Turbine Blade Optimization

This project was built for **ENGIN 26: Three-Dimensional Modeling for Design at UC Berkeley** as a design-and-analysis workflow for wind-turbine blade development. The implementation was constructed from scratch to connect:

- 2D airfoil analysis with XFOIL,
- 3D rotor performance prediction with BEM,
- and multi-objective optimization (NSGA-II) for design tradeoff exploration.

The goal was to demonstrate a full modeling pipeline used in engineering design: define parameters, simulate aerodynamic behavior, optimize competing objectives (power vs. load), and justify the final blade geometry with equations and generated evidence.

This project builds a full wind-turbine aerodynamic optimization model from scratch using:

- `XFOIL` for airfoil polars (`Cl`, `Cd`)
- `BEM` (Blade Element Momentum) for rotor performance
- `NSGA-II` style multi-objective optimization

It optimizes and justifies:

- blade profile (airfoil)
- angle of attack
- twist distribution
- number of blades

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
- `local_sensitivity.csv`: local finite-difference sensitivity of each selected parameter
- `best_geometry.png`: chord and twist distributions
- `best_airfoil_profile.png`: 2D airfoil/profile plot from x,y coordinates
- `pareto_cp_vs_moment.png`: Pareto scatter
- `summary.json`: selected best compromise design
- `report.md`: equation-based justification for profile, AoA, twist, and blade count

How `local_sensitivity.csv` works (brief):

- It perturbs each selected parameter around the final design and re-runs the same XFOIL+BEM physics.
- For continuous parameters (`tip_speed_ratio`, `aoa_deg`, `hub_radius_ratio`, `chord_scale`, `twist_scale`) it uses central finite differences.
- For blade count it uses integer neighboring values.
- For airfoil it switches to each other candidate profile and reports the tradeoff shift.
- Reported derivatives (`dcp_dparam`, `droot_moment_dparam`, `dsolidity_dparam`) quantify how strongly each parameter drives performance and load near the chosen point.
- `impact_tradeoff` ranks which parameter changes most disturb the selected Pareto compromise.

## 4) Equations used (implemented)

- $\lambda = \Omega R / V_{\infty}$
- $\phi = \arctan\!\left(\frac{1-a}{\lambda_r(1+a')}\right)$
- $\alpha = \phi - (\theta + \beta_{\text{pitch}})$
- $C_n = C_l\cos\phi + C_d\sin\phi$
- $C_t = C_l\sin\phi - C_d\cos\phi$
- $a = \frac{1}{\frac{4F\sin^2\phi}{\sigma C_n} + 1}$
- $a' = \frac{1}{\frac{4F\sin\phi\cos\phi}{\sigma C_t} - 1}$
- $dT = 0.5\,\rho W^2 BcC_n\,dr$
- $dQ = 0.5\,\rho W^2 BcC_t r\,dr$
- $C_p = \frac{P}{0.5\,\rho A V_{\infty}^3}, \quad P = \Omega\int dQ$
- $\theta(r) = \phi_{\text{des}}(r) - \alpha_{\text{design}} - \beta_{\text{pitch}}$
- $c(r) \sim \frac{8\pi r\sin\phi_{\text{des}}}{B\,C_{l,\text{des}}\,\lambda_r}$

## 5) Notes

- The model is aerodynamic-first. Structural, fatigue, and manufacturing constraints should be added for final engineering design.
- Configurable ranges are in `configs/default.yaml`.
