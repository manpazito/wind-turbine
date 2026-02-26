from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from wind_turbine.bem import (
    RotorPerformance,
    SectionGeometry,
    SectionResult,
    design_blade_geometry,
    evaluate_rotor,
)
from wind_turbine.config import DesignSpaceConfig, OptimizerConfig, RotorConfig
from wind_turbine.xfoil import XfoilPolarDatabase


@dataclass(frozen=True)
class DesignVector:
    airfoil_idx: int
    blades: int
    tip_speed_ratio: float
    aoa_deg: float
    hub_radius_ratio: float
    chord_scale: float
    twist_scale: float

    def cache_key(self) -> tuple[int, int, float, float, float, float, float]:
        return (
            self.airfoil_idx,
            self.blades,
            round(self.tip_speed_ratio, 6),
            round(self.aoa_deg, 6),
            round(self.hub_radius_ratio, 6),
            round(self.chord_scale, 6),
            round(self.twist_scale, 6),
        )


@dataclass(frozen=True)
class EvaluatedDesign:
    design: DesignVector
    airfoil: str
    sections: list[SectionGeometry]
    section_results: list[SectionResult]
    performance: RotorPerformance

    @property
    def objectives(self) -> tuple[float, float, float]:
        # Minimize: -Cp, root bending moment proxy, solidity
        return (-self.performance.cp, self.performance.root_moment_nm, self.performance.solidity_mean)

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "airfoil": self.airfoil,
            "blades": self.design.blades,
            "tip_speed_ratio": self.design.tip_speed_ratio,
            "aoa_deg": self.design.aoa_deg,
            "hub_radius_ratio": self.design.hub_radius_ratio,
            "chord_scale": self.design.chord_scale,
            "twist_scale": self.design.twist_scale,
            "cp": self.performance.cp,
            "ct": self.performance.ct,
            "power_w": self.performance.power_w,
            "thrust_n": self.performance.thrust_n,
            "torque_nm": self.performance.torque_nm,
            "root_moment_nm": self.performance.root_moment_nm,
            "solidity_mean": self.performance.solidity_mean,
            "obj_neg_cp": -self.performance.cp,
            "obj_root_moment": self.performance.root_moment_nm,
            "obj_solidity": self.performance.solidity_mean,
        }


@dataclass(frozen=True)
class OptimizationOutcome:
    final_population: list[EvaluatedDesign]
    all_evaluations: list[EvaluatedDesign]
    pareto_front: list[EvaluatedDesign]
    best_compromise: EvaluatedDesign


def _dominates(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> bool:
    lhs_arr = np.array(lhs)
    rhs_arr = np.array(rhs)
    return bool(np.all(lhs_arr <= rhs_arr) and np.any(lhs_arr < rhs_arr))


def fast_non_dominated_sort(items: list[EvaluatedDesign]) -> tuple[list[list[int]], np.ndarray]:
    n = len(items)
    dominates_list: list[list[int]] = [[] for _ in range(n)]
    dominated_count = np.zeros(n, dtype=int)
    rank = np.full(n, -1, dtype=int)
    fronts: list[list[int]] = [[]]

    objectives = [it.objectives for it in items]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objectives[p], objectives[q]):
                dominates_list[p].append(q)
            elif _dominates(objectives[q], objectives[p]):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in dominates_list[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1

    return fronts, rank


def crowding_distance(items: list[EvaluatedDesign], front: list[int]) -> dict[int, float]:
    if not front:
        return {}
    if len(front) <= 2:
        return {idx: float("inf") for idx in front}

    distances = {idx: 0.0 for idx in front}
    obj_array = np.array([items[idx].objectives for idx in front], dtype=float)

    for j in range(obj_array.shape[1]):
        order = np.argsort(obj_array[:, j])
        f_sorted = [front[k] for k in order]
        distances[f_sorted[0]] = float("inf")
        distances[f_sorted[-1]] = float("inf")

        min_v = obj_array[order[0], j]
        max_v = obj_array[order[-1], j]
        scale = max(max_v - min_v, 1e-12)

        for k in range(1, len(order) - 1):
            if math.isinf(distances[f_sorted[k]]):
                continue
            prev_v = obj_array[order[k - 1], j]
            next_v = obj_array[order[k + 1], j]
            distances[f_sorted[k]] += float((next_v - prev_v) / scale)
    return distances


def _range_sample(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def random_design(rng: np.random.Generator, design_space: DesignSpaceConfig) -> DesignVector:
    return DesignVector(
        airfoil_idx=int(rng.integers(0, len(design_space.airfoils))),
        blades=int(rng.choice(np.array(design_space.blades_options))),
        tip_speed_ratio=_range_sample(rng, design_space.tip_speed_ratio_range),
        aoa_deg=_range_sample(rng, design_space.aoa_deg_range),
        hub_radius_ratio=_range_sample(rng, design_space.hub_radius_ratio_range),
        chord_scale=_range_sample(rng, design_space.chord_scale_range),
        twist_scale=_range_sample(rng, design_space.twist_scale_range),
    )


def _clip_to_bounds(design: DesignVector, design_space: DesignSpaceConfig) -> DesignVector:
    return DesignVector(
        airfoil_idx=int(max(0, min(design.airfoil_idx, len(design_space.airfoils) - 1))),
        blades=int(min(max(design.blades, min(design_space.blades_options)), max(design_space.blades_options))),
        tip_speed_ratio=float(
            min(max(design.tip_speed_ratio, design_space.tip_speed_ratio_range[0]), design_space.tip_speed_ratio_range[1])
        ),
        aoa_deg=float(min(max(design.aoa_deg, design_space.aoa_deg_range[0]), design_space.aoa_deg_range[1])),
        hub_radius_ratio=float(
            min(
                max(design.hub_radius_ratio, design_space.hub_radius_ratio_range[0]),
                design_space.hub_radius_ratio_range[1],
            )
        ),
        chord_scale=float(
            min(max(design.chord_scale, design_space.chord_scale_range[0]), design_space.chord_scale_range[1])
        ),
        twist_scale=float(
            min(max(design.twist_scale, design_space.twist_scale_range[0]), design_space.twist_scale_range[1])
        ),
    )


def crossover_and_mutate(
    p1: DesignVector,
    p2: DesignVector,
    rng: np.random.Generator,
    design_space: DesignSpaceConfig,
    cfg: OptimizerConfig,
) -> DesignVector:
    if rng.random() < cfg.crossover_probability:
        beta = float(rng.uniform(0.0, 1.0))
        child = DesignVector(
            airfoil_idx=int(p1.airfoil_idx if rng.random() < 0.5 else p2.airfoil_idx),
            blades=int(p1.blades if rng.random() < 0.5 else p2.blades),
            tip_speed_ratio=p1.tip_speed_ratio + beta * (p2.tip_speed_ratio - p1.tip_speed_ratio),
            aoa_deg=p1.aoa_deg + beta * (p2.aoa_deg - p1.aoa_deg),
            hub_radius_ratio=p1.hub_radius_ratio + beta * (p2.hub_radius_ratio - p1.hub_radius_ratio),
            chord_scale=p1.chord_scale + beta * (p2.chord_scale - p1.chord_scale),
            twist_scale=p1.twist_scale + beta * (p2.twist_scale - p1.twist_scale),
        )
    else:
        child = p1

    if rng.random() < cfg.mutation_probability:
        child = DesignVector(
            airfoil_idx=int(rng.integers(0, len(design_space.airfoils))),
            blades=int(rng.choice(np.array(design_space.blades_options))),
            tip_speed_ratio=child.tip_speed_ratio,
            aoa_deg=child.aoa_deg,
            hub_radius_ratio=child.hub_radius_ratio,
            chord_scale=child.chord_scale,
            twist_scale=child.twist_scale,
        )
    if rng.random() < cfg.mutation_probability:
        tsr_span = design_space.tip_speed_ratio_range[1] - design_space.tip_speed_ratio_range[0]
        child = DesignVector(
            airfoil_idx=child.airfoil_idx,
            blades=child.blades,
            tip_speed_ratio=child.tip_speed_ratio + float(rng.normal(0.0, 0.08 * tsr_span)),
            aoa_deg=child.aoa_deg,
            hub_radius_ratio=child.hub_radius_ratio,
            chord_scale=child.chord_scale,
            twist_scale=child.twist_scale,
        )
    if rng.random() < cfg.mutation_probability:
        aoa_span = design_space.aoa_deg_range[1] - design_space.aoa_deg_range[0]
        child = DesignVector(
            airfoil_idx=child.airfoil_idx,
            blades=child.blades,
            tip_speed_ratio=child.tip_speed_ratio,
            aoa_deg=child.aoa_deg + float(rng.normal(0.0, 0.08 * aoa_span)),
            hub_radius_ratio=child.hub_radius_ratio,
            chord_scale=child.chord_scale,
            twist_scale=child.twist_scale,
        )
    if rng.random() < cfg.mutation_probability:
        hub_span = design_space.hub_radius_ratio_range[1] - design_space.hub_radius_ratio_range[0]
        child = DesignVector(
            airfoil_idx=child.airfoil_idx,
            blades=child.blades,
            tip_speed_ratio=child.tip_speed_ratio,
            aoa_deg=child.aoa_deg,
            hub_radius_ratio=child.hub_radius_ratio + float(rng.normal(0.0, 0.08 * hub_span)),
            chord_scale=child.chord_scale,
            twist_scale=child.twist_scale,
        )
    if rng.random() < cfg.mutation_probability:
        c_span = design_space.chord_scale_range[1] - design_space.chord_scale_range[0]
        child = DesignVector(
            airfoil_idx=child.airfoil_idx,
            blades=child.blades,
            tip_speed_ratio=child.tip_speed_ratio,
            aoa_deg=child.aoa_deg,
            hub_radius_ratio=child.hub_radius_ratio,
            chord_scale=child.chord_scale + float(rng.normal(0.0, 0.08 * c_span)),
            twist_scale=child.twist_scale,
        )
    if rng.random() < cfg.mutation_probability:
        t_span = design_space.twist_scale_range[1] - design_space.twist_scale_range[0]
        child = DesignVector(
            airfoil_idx=child.airfoil_idx,
            blades=child.blades,
            tip_speed_ratio=child.tip_speed_ratio,
            aoa_deg=child.aoa_deg,
            hub_radius_ratio=child.hub_radius_ratio,
            chord_scale=child.chord_scale,
            twist_scale=child.twist_scale + float(rng.normal(0.0, 0.08 * t_span)),
        )

    return _clip_to_bounds(child, design_space)


def _tournament_pick(
    population: list[EvaluatedDesign],
    rank: np.ndarray,
    crowding: dict[int, float],
    rng: np.random.Generator,
) -> EvaluatedDesign:
    i, j = int(rng.integers(0, len(population))), int(rng.integers(0, len(population)))
    if rank[i] < rank[j]:
        return population[i]
    if rank[j] < rank[i]:
        return population[j]
    if crowding.get(i, 0.0) > crowding.get(j, 0.0):
        return population[i]
    return population[j]


def _environmental_selection(candidates: list[EvaluatedDesign], population_size: int) -> list[EvaluatedDesign]:
    fronts, _ = fast_non_dominated_sort(candidates)
    selected: list[EvaluatedDesign] = []
    for front in fronts:
        if len(selected) + len(front) <= population_size:
            selected.extend(candidates[idx] for idx in front)
            continue
        distances = crowding_distance(candidates, front)
        ordered = sorted(front, key=lambda idx: distances.get(idx, 0.0), reverse=True)
        remaining = population_size - len(selected)
        selected.extend(candidates[idx] for idx in ordered[:remaining])
        break
    return selected


def evaluate_design(
    design: DesignVector,
    rotor: RotorConfig,
    design_space: DesignSpaceConfig,
    polar_db: XfoilPolarDatabase,
) -> EvaluatedDesign:
    airfoil = design_space.airfoils[design.airfoil_idx]
    sections = design_blade_geometry(
        rotor=rotor,
        design_space=design_space,
        polar_db=polar_db,
        airfoil=airfoil,
        blades=design.blades,
        tip_speed_ratio=design.tip_speed_ratio,
        design_aoa_deg=design.aoa_deg,
        hub_radius_ratio=design.hub_radius_ratio,
        chord_scale=design.chord_scale,
        twist_scale=design.twist_scale,
    )
    section_results, perf = evaluate_rotor(
        rotor=rotor,
        polar_db=polar_db,
        airfoil=airfoil,
        blades=design.blades,
        tip_speed_ratio=design.tip_speed_ratio,
        hub_radius_ratio=design.hub_radius_ratio,
        sections=sections,
    )
    if not math.isfinite(perf.cp):
        perf = RotorPerformance(
            cp=-1.0,
            ct=0.0,
            power_w=0.0,
            thrust_n=0.0,
            torque_nm=0.0,
            root_moment_nm=1e15,
            solidity_mean=1e6,
        )
    return EvaluatedDesign(
        design=design,
        airfoil=airfoil,
        sections=sections,
        section_results=section_results,
        performance=perf,
    )


def run_nsga2(
    rotor: RotorConfig,
    design_space: DesignSpaceConfig,
    optimizer_cfg: OptimizerConfig,
    polar_db: XfoilPolarDatabase,
    seed: int,
) -> OptimizationOutcome:
    rng = np.random.default_rng(seed)

    eval_cache: dict[tuple[int, int, float, float, float, float, float], EvaluatedDesign] = {}

    def get_eval(design: DesignVector) -> EvaluatedDesign:
        key = design.cache_key()
        if key not in eval_cache:
            eval_cache[key] = evaluate_design(design, rotor, design_space, polar_db)
        return eval_cache[key]

    population = [random_design(rng, design_space) for _ in range(optimizer_cfg.population_size)]
    evaluated_population = [get_eval(p) for p in population]
    all_evaluations: list[EvaluatedDesign] = list(evaluated_population)

    for _ in range(optimizer_cfg.generations):
        fronts, rank = fast_non_dominated_sort(evaluated_population)
        crowding: dict[int, float] = {}
        for front in fronts:
            crowding.update(crowding_distance(evaluated_population, front))

        offspring_designs: list[DesignVector] = []
        while len(offspring_designs) < optimizer_cfg.population_size:
            p1 = _tournament_pick(evaluated_population, rank, crowding, rng)
            p2 = _tournament_pick(evaluated_population, rank, crowding, rng)
            child = crossover_and_mutate(
                p1=p1.design,
                p2=p2.design,
                rng=rng,
                design_space=design_space,
                cfg=optimizer_cfg,
            )
            offspring_designs.append(child)

        offspring_eval = [get_eval(x) for x in offspring_designs]
        all_evaluations.extend(offspring_eval)

        combined = evaluated_population + offspring_eval
        evaluated_population = _environmental_selection(combined, optimizer_cfg.population_size)

    fronts, _ = fast_non_dominated_sort(evaluated_population)
    pareto = [evaluated_population[idx] for idx in fronts[0]] if fronts and fronts[0] else evaluated_population
    pareto = _deduplicate_designs(pareto)

    best = choose_best_compromise(pareto)
    return OptimizationOutcome(
        final_population=evaluated_population,
        all_evaluations=_deduplicate_designs(all_evaluations),
        pareto_front=pareto,
        best_compromise=best,
    )


def _deduplicate_designs(items: Iterable[EvaluatedDesign]) -> list[EvaluatedDesign]:
    uniq: dict[tuple[int, int, float, float, float, float, float], EvaluatedDesign] = {}
    for item in items:
        uniq[item.design.cache_key()] = item
    return list(uniq.values())


def choose_best_compromise(pareto: list[EvaluatedDesign]) -> EvaluatedDesign:
    if not pareto:
        raise ValueError("Pareto set is empty.")
    objs = np.array([p.objectives for p in pareto], dtype=float)
    ideal = objs.min(axis=0)
    nadir = objs.max(axis=0)
    norm = (objs - ideal) / (nadir - ideal + 1e-12)
    dist = np.linalg.norm(norm, axis=1)
    return pareto[int(np.argmin(dist))]


def to_dataframe(items: list[EvaluatedDesign]) -> pd.DataFrame:
    return pd.DataFrame([x.to_dict() for x in items]).sort_values("cp", ascending=False).reset_index(drop=True)
