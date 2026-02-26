from __future__ import annotations

import math
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wind_turbine.config import XfoilConfig


@dataclass(frozen=True)
class PolarPoint:
    cl: float
    cd: float


def _looks_like_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _sanitize_airfoil_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip())
    return cleaned.strip("_").lower() or "airfoil"


class XfoilPolarDatabase:
    """XFOIL runner and on-disk polar cache."""

    def __init__(self, config: XfoilConfig, cache_dir: str | Path) -> None:
        self.cfg = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._polar_cache: dict[tuple[str, float], pd.DataFrame] = {}
        self._coords_cache: dict[str, pd.DataFrame] = {}
        self._sorted_re = sorted(float(v) for v in self.cfg.reynolds_bins)

    def prepare(self, airfoils: list[str]) -> None:
        for af in airfoils:
            for re_bin in self._sorted_re:
                self._load_or_generate(af, re_bin)

    def get_design_polar(self, airfoil: str, reynolds: float) -> pd.DataFrame:
        re_bin = self._closest_re_bin(reynolds)
        return self._load_or_generate(airfoil, re_bin)

    def sample(self, airfoil: str, reynolds: float, alpha_deg: float) -> PolarPoint:
        if reynolds <= self._sorted_re[0]:
            return self._sample_single(airfoil, self._sorted_re[0], alpha_deg)
        if reynolds >= self._sorted_re[-1]:
            return self._sample_single(airfoil, self._sorted_re[-1], alpha_deg)

        lo = self._sorted_re[0]
        hi = self._sorted_re[-1]
        for idx in range(1, len(self._sorted_re)):
            if reynolds <= self._sorted_re[idx]:
                lo = self._sorted_re[idx - 1]
                hi = self._sorted_re[idx]
                break

        lo_point = self._sample_single(airfoil, lo, alpha_deg)
        hi_point = self._sample_single(airfoil, hi, alpha_deg)
        w = (reynolds - lo) / max(hi - lo, 1e-12)

        cl = (1.0 - w) * lo_point.cl + w * hi_point.cl
        cd = (1.0 - w) * lo_point.cd + w * hi_point.cd
        return PolarPoint(cl=float(cl), cd=float(max(cd, 1e-5)))

    def get_airfoil_coordinates(self, airfoil: str) -> pd.DataFrame:
        if airfoil in self._coords_cache:
            return self._coords_cache[airfoil]

        cache_file = self.cache_dir / f"{_sanitize_airfoil_name(airfoil)}_coords.csv"
        if cache_file.exists():
            coords = pd.read_csv(cache_file)
        else:
            coords = self._run_xfoil_coordinates(airfoil)
            coords.to_csv(cache_file, index=False)

        coords = coords.reset_index(drop=True)
        if coords.empty:
            raise RuntimeError(f"No coordinates found for airfoil '{airfoil}'.")
        self._coords_cache[airfoil] = coords
        return coords

    def _sample_single(self, airfoil: str, re_bin: float, alpha_deg: float) -> PolarPoint:
        polar = self._load_or_generate(airfoil, re_bin)
        alpha = polar["alpha"].to_numpy()
        cl_data = polar["cl"].to_numpy()
        cd_data = polar["cd"].to_numpy()
        cl = float(np.interp(alpha_deg, alpha, cl_data, left=cl_data[0], right=cl_data[-1]))
        cd = float(np.interp(alpha_deg, alpha, cd_data, left=cd_data[0], right=cd_data[-1]))
        return PolarPoint(cl=cl, cd=max(cd, 1e-5))

    def _cache_file(self, airfoil: str, re_bin: float) -> Path:
        token = _sanitize_airfoil_name(airfoil)
        return self.cache_dir / f"{token}_re_{int(round(re_bin))}.csv"

    def _closest_re_bin(self, reynolds: float) -> float:
        log_re = math.log(max(reynolds, 1.0))
        return min(self._sorted_re, key=lambda v: abs(math.log(v) - log_re))

    def _load_or_generate(self, airfoil: str, re_bin: float) -> pd.DataFrame:
        key = (airfoil, float(re_bin))
        if key in self._polar_cache:
            return self._polar_cache[key]

        cache_file = self._cache_file(airfoil, re_bin)
        if cache_file.exists():
            polar = pd.read_csv(cache_file)
        else:
            polar = self._run_xfoil(airfoil=airfoil, reynolds=re_bin)
            polar.to_csv(cache_file, index=False)
        polar = polar.sort_values("alpha").drop_duplicates("alpha").reset_index(drop=True)
        if polar.empty:
            raise RuntimeError(f"XFOIL returned empty polar for {airfoil} @ Re={re_bin:.0f}")
        self._polar_cache[key] = polar
        return polar

    def _run_xfoil(self, airfoil: str, reynolds: float) -> pd.DataFrame:
        with tempfile.TemporaryDirectory(prefix="xfoil_run_") as tmpdir:
            run_dir = Path(tmpdir)
            polar_path = run_dir / "polar.out"
            command_script = self._build_xfoil_script(
                airfoil=airfoil,
                reynolds=reynolds,
                polar_filename=polar_path.name,
            )
            proc = subprocess.run(
                [self.cfg.executable],
                input=command_script,
                text=True,
                capture_output=True,
                cwd=run_dir,
                timeout=self.cfg.timeout_s,
                check=False,
            )
            if proc.returncode != 0 and not polar_path.exists():
                stderr_excerpt = proc.stderr.strip().splitlines()[-8:]
                raise RuntimeError(
                    f"XFOIL failed for airfoil={airfoil}, Re={reynolds:.0f}. "
                    f"Exit={proc.returncode}. Stderr tail={stderr_excerpt}"
                )
            if not polar_path.exists():
                stdout_excerpt = proc.stdout.strip().splitlines()[-12:]
                raise RuntimeError(
                    f"XFOIL did not write polar file for airfoil={airfoil}, Re={reynolds:.0f}. "
                    f"Stdout tail={stdout_excerpt}"
                )

            polar = self._parse_polar_file(polar_path)
            if len(polar) < 6:
                raise RuntimeError(
                    f"Not enough valid polar points for {airfoil} @ Re={reynolds:.0f}. "
                    f"Points={len(polar)}"
                )
            return polar

    def _run_xfoil_coordinates(self, airfoil: str) -> pd.DataFrame:
        with tempfile.TemporaryDirectory(prefix="xfoil_coords_") as tmpdir:
            run_dir = Path(tmpdir)
            coords_path = run_dir / "coords.dat"
            script = self._build_coords_script(airfoil=airfoil, coords_filename=coords_path.name)
            proc = subprocess.run(
                [self.cfg.executable],
                input=script,
                text=True,
                capture_output=True,
                cwd=run_dir,
                timeout=self.cfg.timeout_s,
                check=False,
            )
            if proc.returncode != 0 and not coords_path.exists():
                stderr_excerpt = proc.stderr.strip().splitlines()[-8:]
                raise RuntimeError(
                    f"XFOIL failed while exporting coordinates for airfoil={airfoil}. "
                    f"Exit={proc.returncode}. Stderr tail={stderr_excerpt}"
                )
            if not coords_path.exists():
                stdout_excerpt = proc.stdout.strip().splitlines()[-12:]
                raise RuntimeError(
                    f"XFOIL did not write coordinate file for airfoil={airfoil}. "
                    f"Stdout tail={stdout_excerpt}"
                )
            coords = self._parse_coords_file(coords_path)
            if len(coords) < 20:
                raise RuntimeError(
                    f"Not enough coordinate points for airfoil={airfoil}. "
                    f"Points={len(coords)}"
                )
            return coords

    def _build_xfoil_script(self, airfoil: str, reynolds: float, polar_filename: str) -> str:
        af = airfoil.strip().upper().replace("NACA", "").strip()
        if af.isdigit():
            load_line = f"NACA {af}"
        else:
            af_path = Path(airfoil).expanduser().resolve()
            load_line = f"LOAD {af_path}"

        return (
            f"{load_line}\n"
            "PANE\n"
            "OPER\n"
            f"VISC {int(round(reynolds))}\n"
            "MACH 0.0\n"
            "VPAR\n"
            f"N {self.cfg.ncrit}\n"
            "\n"
            f"ITER {int(self.cfg.max_iter)}\n"
            "PACC\n"
            f"{polar_filename}\n"
            "\n"
            f"ASEQ {self.cfg.alpha_start_deg:.3f} {self.cfg.alpha_end_deg:.3f} {self.cfg.alpha_step_deg:.3f}\n"
            "PACC\n"
            "\n"
            "QUIT\n"
        )

    def _build_coords_script(self, airfoil: str, coords_filename: str) -> str:
        af = airfoil.strip().upper().replace("NACA", "").strip()
        if af.isdigit():
            load_line = f"NACA {af}"
        else:
            af_path = Path(airfoil).expanduser().resolve()
            load_line = f"LOAD {af_path}"
        return (
            f"{load_line}\n"
            "PANE\n"
            "PSAV\n"
            f"{coords_filename}\n"
            "QUIT\n"
        )

    @staticmethod
    def _parse_polar_file(path: Path) -> pd.DataFrame:
        rows: list[tuple[float, float, float]] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            cols = line.strip().split()
            if len(cols) < 3:
                continue
            if not (_looks_like_number(cols[0]) and _looks_like_number(cols[1]) and _looks_like_number(cols[2])):
                continue
            alpha = float(cols[0])
            cl = float(cols[1])
            cd = float(cols[2])
            if not math.isfinite(alpha) or not math.isfinite(cl) or not math.isfinite(cd):
                continue
            rows.append((alpha, cl, max(cd, 1e-5)))
        return pd.DataFrame(rows, columns=["alpha", "cl", "cd"])

    @staticmethod
    def _parse_coords_file(path: Path) -> pd.DataFrame:
        rows: list[tuple[float, float]] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            cols = line.strip().split()
            if len(cols) < 2:
                continue
            if not (_looks_like_number(cols[0]) and _looks_like_number(cols[1])):
                continue
            x = float(cols[0])
            y = float(cols[1])
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            rows.append((x, y))
        return pd.DataFrame(rows, columns=["x", "y"])
