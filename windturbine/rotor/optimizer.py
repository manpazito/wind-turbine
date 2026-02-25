from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from windturbine.config import OptimizerConfig, RotorConfig, SurrogateConfig
from windturbine.rotor.bem import (
    RotorPerformance,
    RotorSection,
    SectionResult,
    design_rotor_sections,
    evaluate_rotor,
)
from windturbine.rotor.polars import PolarModel

LOGGER = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    blades: int
    aoa_deg: float
    tip_speed_ratio: float
    hub_radius_ratio: float
    max_chord_m: float
    sections: list[RotorSection]
    section_results: list[SectionResult]
    performance: RotorPerformance
    evaluated_candidates: int
    surrogate_model: object | None = None
    pareto: pd.DataFrame | None = None
    surrogate_metrics: dict[str, float] | None = None


@dataclass
class _EvalMetrics:
    cp: float
    root_moment_proxy: float
    solidity: float
    noise_proxy: float
    blade_area: float
    sections: list[RotorSection]
    section_results: list[SectionResult]
    performance: RotorPerformance


class MLPRegressor:
    """Small numpy MLP for multi-output regression."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.2, size=(in_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = rng.normal(0.0, 0.2, size=(hidden_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)
        self.w3 = rng.normal(0.0, 0.2, size=(hidden_dim, out_dim))
        self.b3 = np.zeros(out_dim)
        self.x_mean = np.zeros(in_dim)
        self.x_std = np.ones(in_dim)
        self.y_mean = np.zeros(out_dim)
        self.y_std = np.ones(out_dim)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z1 = x @ self.w1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = self._relu(z2)
        yhat = a2 @ self.w3 + self.b3
        return z1, a1, z2, a2, yhat

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 300,
        lr: float = 2e-3,
        l2: float = 1e-5,
    ) -> dict[str, float]:
        self.x_mean = x_train.mean(axis=0)
        self.x_std = np.where(x_train.std(axis=0) > 1e-9, x_train.std(axis=0), 1.0)
        self.y_mean = y_train.mean(axis=0)
        self.y_std = np.where(y_train.std(axis=0) > 1e-9, y_train.std(axis=0), 1.0)

        xtr = (x_train - self.x_mean) / self.x_std
        ytr = (y_train - self.y_mean) / self.y_std
        xva = (x_val - self.x_mean) / self.x_std
        yva = (y_val - self.y_mean) / self.y_std

        n = xtr.shape[0]
        for _ in range(epochs):
            z1, a1, z2, a2, yhat = self._forward(xtr)
            err = (yhat - ytr)
            grad_y = (2.0 / n) * err

            gw3 = a2.T @ grad_y + l2 * self.w3
            gb3 = grad_y.sum(axis=0)

            ga2 = grad_y @ self.w3.T
            gz2 = ga2 * (z2 > 0.0)
            gw2 = a1.T @ gz2 + l2 * self.w2
            gb2 = gz2.sum(axis=0)

            ga1 = gz2 @ self.w2.T
            gz1 = ga1 * (z1 > 0.0)
            gw1 = xtr.T @ gz1 + l2 * self.w1
            gb1 = gz1.sum(axis=0)

            self.w3 -= lr * gw3
            self.b3 -= lr * gb3
            self.w2 -= lr * gw2
            self.b2 -= lr * gb2
            self.w1 -= lr * gw1
            self.b1 -= lr * gb1

        pred_val = self.predict(x_val)
        abs_err = np.abs(pred_val - y_val)
        mse = np.mean((pred_val - y_val) ** 2, axis=0)
        var = np.var(y_val, axis=0) + 1e-9
        r2 = 1.0 - (mse / var)
        return {
            "val_mae_cp": float(abs_err[:, 0].mean()),
            "val_mae_root_moment": float(abs_err[:, 1].mean()),
            "val_mae_solidity": float(abs_err[:, 2].mean()),
            "val_mae_noise": float(abs_err[:, 3].mean()),
            "val_r2_cp": float(r2[0]),
            "val_r2_root_moment": float(r2[1]),
            "val_r2_solidity": float(r2[2]),
            "val_r2_noise": float(r2[3]),
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        xz = (x - self.x_mean) / self.x_std
        *_, yhat = self._forward(xz)
        return yhat * self.y_std + self.y_mean


def _evaluate_design(
    rotor_cfg: RotorConfig,
    polar: PolarModel,
    blades: int,
    aoa_deg: float,
    tsr: float,
    hub_ratio: float,
    max_chord_m: float,
) -> _EvalMetrics:
    sections = design_rotor_sections(
        radius_m=rotor_cfg.radius_m,
        hub_radius_ratio=hub_ratio,
        n_sections=rotor_cfg.n_sections,
        blades=blades,
        tip_speed_ratio=tsr,
        design_aoa_deg=aoa_deg,
        pitch_deg=rotor_cfg.pitch_deg,
        polar=polar,
    )
    clipped_sections = [
        RotorSection(
            r_over_R=s.r_over_R,
            r_m=s.r_m,
            chord_m=float(min(s.chord_m, max_chord_m)),
            twist_deg=s.twist_deg,
        )
        for s in sections
    ]

    section_results, performance = evaluate_rotor(
        sections=clipped_sections,
        blades=blades,
        radius_m=rotor_cfg.radius_m,
        hub_radius_ratio=hub_ratio,
        tip_speed_ratio=tsr,
        wind_speed_ms=rotor_cfg.wind_speed_ms,
        pitch_deg=rotor_cfg.pitch_deg,
        air_density=rotor_cfg.air_density,
        dynamic_viscosity=rotor_cfg.dynamic_viscosity,
        polar=polar,
    )
    if not section_results:
        raise RuntimeError("Empty section results")

    dr = (rotor_cfg.radius_m - hub_ratio * rotor_cfg.radius_m) / max(rotor_cfg.n_sections, 1)
    root_radius = hub_ratio * rotor_cfg.radius_m
    root_moment = 0.0
    for sec in section_results:
        arm = max(sec.r_m - root_radius, 0.0)
        root_moment += sec.dthrust_n * arm

    blade_area = float(blades * sum(sec.chord_m * dr for sec in section_results))
    solidity = float(np.mean([sec.local_solidity for sec in section_results]))
    tip_speed = tsr * rotor_cfg.wind_speed_ms
    noise_proxy = float((tip_speed**2) * (1.0 + 2.0 * solidity))

    return _EvalMetrics(
        cp=float(performance.cp),
        root_moment_proxy=float(root_moment),
        solidity=solidity,
        noise_proxy=noise_proxy,
        blade_area=blade_area,
        sections=clipped_sections,
        section_results=section_results,
        performance=performance,
    )


def _sample_designs(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    b = rng.choice(np.array([2.0, 3.0, 4.0]), size=n)
    tsr = rng.uniform(4.0, 10.0, size=n)
    alpha = rng.uniform(2.0, 10.0, size=n)
    hub = rng.uniform(0.15, 0.30, size=n)
    max_chord_ratio = rng.uniform(0.04, 0.14, size=n)
    return np.column_stack([b, tsr, alpha, hub, max_chord_ratio])


def _build_dataset(
    rotor_cfg: RotorConfig,
    polar: PolarModel,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = _sample_designs(n_samples, seed)
    y_rows: list[list[float]] = []
    for row in x:
        blades = int(round(row[0]))
        tsr, alpha, hub, chord_ratio = float(row[1]), float(row[2]), float(row[3]), float(row[4])
        max_chord_m = chord_ratio * rotor_cfg.radius_m
        metrics = _evaluate_design(
            rotor_cfg=rotor_cfg,
            polar=polar,
            blades=blades,
            aoa_deg=alpha,
            tsr=tsr,
            hub_ratio=hub,
            max_chord_m=max_chord_m,
        )
        y_rows.append([metrics.cp, metrics.root_moment_proxy, metrics.solidity, metrics.noise_proxy])
    y = np.asarray(y_rows, dtype=float)
    x[:, 0] = np.round(x[:, 0])
    return x, y


def _objectives_from_pred(y_pred: np.ndarray, x: np.ndarray, radius_m: float) -> np.ndarray:
    cp = y_pred[:, 0]
    root_moment = y_pred[:, 1]
    blade_area = x[:, 0] * (x[:, 4] * radius_m) * radius_m
    return np.column_stack([-cp, root_moment, blade_area])


def _non_dominated_sort(objs: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    n = objs.shape[0]
    dominates: list[list[int]] = [[] for _ in range(n)]
    dominated_count = np.zeros(n, dtype=int)
    rank = np.full(n, -1, dtype=int)
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            p_le_q = np.all(objs[p] <= objs[q])
            p_lt_q = np.any(objs[p] < objs[q])
            q_le_p = np.all(objs[q] <= objs[p])
            q_lt_p = np.any(objs[q] < objs[p])
            if p_le_q and p_lt_q:
                dominates[p].append(q)
            elif q_le_p and q_lt_p:
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in dominates[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1
    return fronts, rank


def _crowding_distance(front: list[int], objs: np.ndarray) -> np.ndarray:
    dist = np.zeros(len(front), dtype=float)
    if len(front) <= 2:
        dist[:] = np.inf
        return dist
    front_objs = objs[front]
    m = front_objs.shape[1]
    for k in range(m):
        order = np.argsort(front_objs[:, k])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        span = front_objs[order[-1], k] - front_objs[order[0], k]
        if span <= 1e-12:
            continue
        for i in range(1, len(front) - 1):
            dist[order[i]] += (front_objs[order[i + 1], k] - front_objs[order[i - 1], k]) / span
    return dist


def _repair(pop: np.ndarray) -> np.ndarray:
    out = pop.copy()
    out[:, 0] = np.clip(np.round(out[:, 0]), 2.0, 4.0)
    out[:, 1] = np.clip(out[:, 1], 4.0, 10.0)
    out[:, 2] = np.clip(out[:, 2], 2.0, 10.0)
    out[:, 3] = np.clip(out[:, 3], 0.15, 0.30)
    out[:, 4] = np.clip(out[:, 4], 0.04, 0.14)
    return out


def _nsga2_optimize(
    surrogate: MLPRegressor,
    rotor_cfg: RotorConfig,
    seed: int,
    population_size: int = 120,
    generations: int = 45,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pop = _sample_designs(population_size, seed)

    def evaluate_population(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y_pred = surrogate.predict(x)
        return y_pred, _objectives_from_pred(y_pred, x, rotor_cfg.radius_m)

    y_pred, objs = evaluate_population(pop)
    for _ in range(generations):
        fronts, rank = _non_dominated_sort(objs)
        crowd = np.zeros(pop.shape[0], dtype=float)
        for front in fronts:
            if not front:
                continue
            crowd_front = _crowding_distance(front, objs)
            for idx, val in zip(front, crowd_front):
                crowd[idx] = val

        def tournament() -> int:
            i, j = int(rng.integers(0, pop.shape[0])), int(rng.integers(0, pop.shape[0]))
            if rank[i] < rank[j]:
                return i
            if rank[j] < rank[i]:
                return j
            return i if crowd[i] >= crowd[j] else j

        offspring = []
        for _ in range(population_size // 2):
            p1 = pop[tournament()]
            p2 = pop[tournament()]
            beta = rng.uniform(0.2, 0.8, size=p1.shape[0])
            c1 = beta * p1 + (1.0 - beta) * p2
            c2 = beta * p2 + (1.0 - beta) * p1
            c1 += rng.normal(0.0, [0.1, 0.2, 0.2, 0.01, 0.005], size=c1.shape)
            c2 += rng.normal(0.0, [0.1, 0.2, 0.2, 0.01, 0.005], size=c2.shape)
            offspring.append(c1)
            offspring.append(c2)
        offspring_arr = _repair(np.asarray(offspring, dtype=float))

        combined = np.vstack([pop, offspring_arr])
        combined_y, combined_objs = evaluate_population(combined)
        fronts, _ = _non_dominated_sort(combined_objs)

        new_pop_idx: list[int] = []
        for front in fronts:
            if len(new_pop_idx) + len(front) <= population_size:
                new_pop_idx.extend(front)
            else:
                crowd_front = _crowding_distance(front, combined_objs)
                order = np.argsort(crowd_front)[::-1]
                needed = population_size - len(new_pop_idx)
                new_pop_idx.extend([front[i] for i in order[:needed]])
                break

        idx = np.asarray(new_pop_idx, dtype=int)
        pop = combined[idx]
        y_pred = combined_y[idx]
        objs = combined_objs[idx]

    return pop, y_pred


def reoptimize_with_constraints(
    surrogate: MLPRegressor,
    rotor_cfg: RotorConfig,
    constraints: dict[str, float],
    seed: int = 42,
) -> pd.DataFrame:
    pop, y_pred = _nsga2_optimize(surrogate, rotor_cfg, seed=seed)
    df = pd.DataFrame(
        {
            "B": pop[:, 0].round().astype(int),
            "TSR": pop[:, 1],
            "alpha_opt_deg": pop[:, 2],
            "r0_over_R": pop[:, 3],
            "max_chord_ratio": pop[:, 4],
            "cp_pred": y_pred[:, 0],
            "root_moment_pred": y_pred[:, 1],
            "solidity_pred": y_pred[:, 2],
            "noise_pred": y_pred[:, 3],
        }
    )
    if "max_noise" in constraints:
        df = df[df["noise_pred"] <= constraints["max_noise"]]
    if "min_cp" in constraints:
        df = df[df["cp_pred"] >= constraints["min_cp"]]
    return df.sort_values(["cp_pred", "root_moment_pred"], ascending=[False, True]).reset_index(drop=True)


def optimize_rotor(
    rotor_cfg: RotorConfig,
    optimizer_cfg: OptimizerConfig,
    polar: PolarModel,
    surrogate_cfg: SurrogateConfig,
    output_dir: Path | None = None,
) -> OptimizationResult:
    output_dir = output_dir or Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = int(getattr(surrogate_cfg, "n_samples", 2000))
    seed = int(getattr(surrogate_cfg, "seed", 42))
    x, y = _build_dataset(rotor_cfg=rotor_cfg, polar=polar, n_samples=n_samples, seed=seed)

    split = int(0.8 * x.shape[0])
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]

    model = MLPRegressor(in_dim=5, hidden_dim=48, out_dim=4, seed=seed)
    metrics = model.fit(x_train, y_train, x_val, y_val, epochs=320, lr=2e-3, l2=1e-5)

    (output_dir / "surrogate_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pop_size = int(getattr(surrogate_cfg, "population_size", 120))
    generations = int(getattr(surrogate_cfg, "generations", 45))
    pop, y_pred = _nsga2_optimize(
        surrogate=model,
        rotor_cfg=rotor_cfg,
        seed=seed + 1,
        population_size=pop_size,
        generations=generations,
    )

    objs = _objectives_from_pred(y_pred, pop, rotor_cfg.radius_m)
    fronts, _ = _non_dominated_sort(objs)
    first_front = fronts[0] if fronts else list(range(pop.shape[0]))

    pareto_rows = []
    for idx in first_front:
        pareto_rows.append(
            {
                "B": int(round(pop[idx, 0])),
                "TSR": float(pop[idx, 1]),
                "alpha_opt_deg": float(pop[idx, 2]),
                "r0_over_R": float(pop[idx, 3]),
                "max_chord_ratio": float(pop[idx, 4]),
                "cp_pred": float(y_pred[idx, 0]),
                "root_moment_pred": float(y_pred[idx, 1]),
                "solidity_pred": float(y_pred[idx, 2]),
                "noise_pred": float(y_pred[idx, 3]),
                "obj_neg_cp": float(objs[idx, 0]),
                "obj_root_moment": float(objs[idx, 1]),
                "obj_area": float(objs[idx, 2]),
            }
        )
    pareto_df = pd.DataFrame(pareto_rows).sort_values(
        ["obj_neg_cp", "obj_root_moment", "obj_area"], ascending=[True, True, True]
    )
    pareto_path = output_dir / "pareto.csv"
    pareto_df.to_csv(pareto_path, index=False)

    top_k = int(getattr(surrogate_cfg, "top_k_candidates", 8))
    validate_df = pareto_df.head(max(top_k, 1)).copy()

    validated: list[dict[str, float]] = []
    best_metrics: _EvalMetrics | None = None
    best_design: dict[str, float] | None = None
    best_score = -np.inf

    for _, row in validate_df.iterrows():
        b = int(row["B"])
        tsr = float(row["TSR"])
        alpha = float(row["alpha_opt_deg"])
        hub = float(row["r0_over_R"])
        max_chord_m = float(row["max_chord_ratio"] * rotor_cfg.radius_m)
        met = _evaluate_design(
            rotor_cfg=rotor_cfg,
            polar=polar,
            blades=b,
            aoa_deg=alpha,
            tsr=tsr,
            hub_ratio=hub,
            max_chord_m=max_chord_m,
        )
        validated.append(
            {
                "B": b,
                "TSR": tsr,
                "alpha_opt_deg": alpha,
                "r0_over_R": hub,
                "max_chord_ratio": row["max_chord_ratio"],
                "cp_true": met.cp,
                "root_moment_true": met.root_moment_proxy,
                "solidity_true": met.solidity,
                "noise_true": met.noise_proxy,
                "blade_area_true": met.blade_area,
            }
        )
        score = met.cp - 1e-6 * met.root_moment_proxy - 0.001 * met.blade_area
        if score > best_score:
            best_score = score
            best_metrics = met
            best_design = {
                "B": float(b),
                "TSR": tsr,
                "alpha_opt_deg": alpha,
                "r0_over_R": hub,
                "max_chord_ratio": float(row["max_chord_ratio"]),
            }

    if best_metrics is None or best_design is None:
        raise RuntimeError("No valid design after BEM validation")

    top_sections = pd.DataFrame(
        [
            {
                "r_over_R": s.r_over_R,
                "r_m": s.r_m,
                "chord_m": s.chord_m,
                "twist_deg": s.twist_deg,
                "aoa_deg": s.aoa_deg,
                "phi_deg": s.phi_deg,
                "cl": s.cl,
                "cd": s.cd,
                "reynolds": s.reynolds,
                "local_solidity": s.local_solidity,
            }
            for s in best_metrics.section_results
        ]
    )
    top_sections.to_csv(output_dir / "top_design_rotor_sections.csv", index=False)

    return OptimizationResult(
        blades=int(best_design["B"]),
        aoa_deg=float(best_design["alpha_opt_deg"]),
        tip_speed_ratio=float(best_design["TSR"]),
        hub_radius_ratio=float(best_design["r0_over_R"]),
        max_chord_m=float(best_design["max_chord_ratio"] * rotor_cfg.radius_m),
        sections=best_metrics.sections,
        section_results=best_metrics.section_results,
        performance=best_metrics.performance,
        evaluated_candidates=int(len(validated)),
        surrogate_model=model,
        pareto=pareto_df,
        surrogate_metrics=metrics,
    )
