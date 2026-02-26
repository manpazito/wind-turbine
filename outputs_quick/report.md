# Wind Turbine Optimization Report

This run follows the multi-objective spirit of the provided paper, but uses a wind-turbine workflow:
XFOIL polars + BEM simulation + NSGA-II search on blade profile, AoA, twist scaling, TSR, and blade count.

## Core Equations Used

1. Tip speed ratio:
   - `lambda = Omega * R / V_inf`
2. Inflow angle and angle of attack:
   - `phi = atan((1-a) / (lambda_r * (1+a')))`
   - `alpha = phi - (theta + beta_pitch)`
3. Force coefficients:
   - `C_n = C_l cos(phi) + C_d sin(phi)`
   - `C_t = C_l sin(phi) - C_d cos(phi)`
4. Induction factors (BEM form):
   - `a = 1 / ( (4F sin^2(phi))/(sigma C_n) + 1 )`
   - `a' = 1 / ( (4F sin(phi)cos(phi))/(sigma C_t) - 1 )`
5. Section loads and rotor power:
   - `dT = 0.5 rho W^2 B c C_n dr`
   - `dQ = 0.5 rho W^2 B c C_t r dr`
   - `Cp = P / (0.5 rho A V_inf^3),  P = Omega * integral(dQ)`
6. Geometry synthesis equations:
   - `phi_des(r) ~= (2/3) atan(1/lambda_r)`
   - `theta(r) = phi_des(r) - alpha_design - beta_pitch`
   - `c(r) ~ 8 pi r sin(phi_des)/(B C_l,des lambda_r)`

## Optimized Design (Best Compromise From Pareto)

- Airfoil profile: `NACA 23012`
- Blade number: `2`
- Design AoA: `4.704 deg`
- Tip speed ratio: `5.859`
- Hub radius ratio: `0.1896`
- Twist scale: `1.0247`
- Chord scale: `0.9443`
- Achieved Cp: `0.3962`
- Root moment: `3065062.62 N.m`
- Mean solidity: `0.07154`

## Justification

- Blade profile choice (`NACA 23012`): selected on the Pareto front where it balances high Cp and moderate root load.
- Angle of attack: optimized AoA `4.704 deg`; XFOIL at Re≈10304574 gives peak Cl/Cd around `8.000 deg` (Cl/Cd≈122.535), so chosen AoA is in a high-efficiency aerodynamic band.
- Twist: optimized from `theta(r)=phi_des-alpha_design-beta_pitch`; resulting twist decreases from `20.159 deg` at root to `2.070 deg` near tip, which is physically consistent with lower inflow angle at larger radius.
- Blade number: `2` emerged from multi-objective trade-off; the blade-count means on Pareto solutions are reported below for transparency.

## Pareto Mean Performance by Blade Count

- B=2: Cp=0.3894, RootMoment=3196802.76 N.m, Solidity=0.06908
- B=3: Cp=0.4391, RootMoment=3731738.44 N.m, Solidity=0.10430
- B=4: Cp=0.3673, RootMoment=3757655.75 N.m, Solidity=0.13940

## Pareto Mean Performance by Airfoil

- NACA 23012: Cp=0.4187, RootMoment=3515402.52 N.m, Solidity=0.09228
- NACA 2412: Cp=0.3415, RootMoment=3153356.48 N.m, Solidity=0.08137

## Notes

- This is an aerodynamic optimization model. Structural and fatigue constraints should be added before manufacturing decisions.
- The workflow is deterministic for a fixed random seed and config file.
