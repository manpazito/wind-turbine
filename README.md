# Wind Turbine Design Toolkit

End-to-end wind turbine conceptual design workflow using:
- Pritchard 11-parameter blade profile generation
- BEM (Blade Element Momentum) rotor analysis
- Surrogate model + NSGA-II style multi-objective optimization

## 1) Prerequisites

- Windows PowerShell
- Python 3.10+ (3.11 recommended)

## 2) Get the code and open project folder

```powershell
git clone <your-repo-url>
cd wind-turbine
```

## 3) (Optional) Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 4) Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib pyyaml pytest
```

## 5) Configure inputs

Edit:
- `configs/default.yaml`

Main groups:
- `airfoil`: 11 Pritchard parameters
- `rotor`: rotor radius, wind speed, density, sections
- `polar`: lift/drag model settings
- `surrogate`: dataset size, seed, optimizer settings

## 6) Run the full design pipeline

```powershell
python -m windturbine.design --config configs/default.yaml
```

## 7) Check outputs

Generated in `.\outputs\`:
- `airfoil_coords.csv`
- `airfoil_plot.png`
- `rotor_sections.csv`
- `pareto.csv`
- `surrogate_metrics.json`
- `top_design_rotor_sections.csv`
- `summary.json`
- `report.md`
- `twist_distribution.png`

## 8) Run tests

```powershell
python -m pytest -q
```

## Notes

- `TurbineBladeGen.py` is loaded from:
  `11-Parameters-Turbine-Blade-Generator/TurbineBladeGen.py`
- Outputs are always written to `.\outputs\`.

## Acknowledgments

Special thanks and credit to **David Poves** for the
**11-Parameters Turbine Blade Generator**, which this project uses for
Pritchard-based blade profile generation:
- `11-Parameters-Turbine-Blade-Generator`

## BSD 2-Clause License

This project is licensed under the BSD 2-Clause License. See:
- `LICENSE`

## Third-Party Notice

The included `11-Parameters-Turbine-Blade-Generator`
is by **David Poves** and contains third-party code under its own license (MIT). See:
- `11-Parameters-Turbine-Blade-Generator/LICENSE`

## References

- Li, L., Zhang, W., Li, Y., Jiang, C., & Wang, Y. (2023).
  *Multi-objective optimization of turbine blade profiles based on multi-agent reinforcement learning*.
  **Energy Conversion and Management, 297**, 117637.
  https://doi.org/10.1016/j.enconman.2023.117637
