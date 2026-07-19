"""Seeded Python reconstruction of available HGALO operators plus final coordination.

The restored MATLAB source exposes HGALO, not an independent CA-HGALO entrypoint.
This module therefore keeps that provenance explicit: it ports the available
optimizer operators and applies the separately validated Python coordinator to
the final trajectory set. It must not be reported as MATLAB-verified CA-HGALO.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from multiuav.core.models import EvaluationResult, Scenario, Trajectory
from multiuav.evaluation.multi_uav import MultiUAVEvaluator
from multiuav.expert.alo import alo_local_exploitation
from multiuav.expert.coordinator import ConflictAwareCoordinator, CoordinationResult
from multiuav.expert.encoding import (
    build_planning_problem,
    chaotic_initialize,
    decode_candidate,
    project_candidate,
    repair_candidate,
)
from multiuav.expert.gwo import gwo_guided_update

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HGALOConfig:
    """MATLAB HGALO defaults with an explicit reproducibility seed."""

    pop_size: int = 45
    max_iter: int = 55
    seed: int = 20260716
    use_chaos_init: bool = True
    use_levy_flight: bool = True
    use_elite_local_search: bool = True
    use_opposition_learning: bool = True
    use_feasibility_repair: bool = True
    use_gwo_guidance: bool = True
    use_alo_local: bool = True
    local_prob: float = 0.3
    top_k_ratio: float = 0.2
    local_scale0: float = 0.05
    local_decay: float = 2.0
    levy_beta: float = 1.5
    elite_preserve: bool = True
    greedy_selection: bool = True
    use_deb_feasibility_rules: bool = False
    use_feasibility_preserving_mutation: bool = False
    adaptive_threat_penalty: bool = False
    adaptive_threat_alpha: float = 0.8
    adaptive_threat_min_scale: float = 0.8
    adaptive_threat_max_scale: float = 6.0

    def __post_init__(self) -> None:
        if self.pop_size < 3 or self.max_iter < 1:
            raise ValueError("HGALO requires pop_size >= 3 and max_iter >= 1.")


@dataclass(frozen=True)
class CandidateEvaluation:
    """One candidate evaluated exclusively through the shared Python evaluator."""

    candidate: FloatArray
    trajectories: tuple[Trajectory, ...]
    evaluation: EvaluationResult
    violation: float

    @property
    def cost(self) -> float:
        """Expose total cost for MATLAB-style population sorting."""
        return self.evaluation.total_cost

    @property
    def success(self) -> bool:
        """Expose strict feasibility for Deb-rule ordering."""
        return self.evaluation.strict_success


@dataclass(frozen=True)
class GenerationStat:
    """Per-generation convergence and feasibility summary."""

    generation: int
    best_cost: float
    feasible_ratio: float
    best_violation: float
    threat_failure_ratio: float


@dataclass(frozen=True)
class HGALOResult:
    """Full optimizer, coordination, logging, and convergence result for one seed."""

    algorithm: str
    provenance: str
    seed: int
    best_candidate: FloatArray
    raw_best_trajectories: tuple[Trajectory, ...]
    best_trajectories: tuple[Trajectory, ...]
    optimizer_evaluation: EvaluationResult
    coordination: CoordinationResult
    convergence_history: FloatArray
    generation_stats: tuple[GenerationStat, ...]


class HGALOPlanner:
    """Port of available MATLAB HGALO evolution with final conflict coordination."""

    def __init__(self, scenario: Scenario, num_waypoints: int, config: HGALOConfig) -> None:
        self.scenario = scenario
        self.problem = build_planning_problem(scenario, num_waypoints)
        self.config = config
        self.evaluator = MultiUAVEvaluator(scenario)

    def run(self) -> HGALOResult:
        """Run seeded greedy HGALO evolution and coordinate the global elite once."""
        rng = np.random.default_rng(self.config.seed)
        population = self._initial_population(rng)
        best: CandidateEvaluation | None = None
        history = np.empty(self.config.max_iter, dtype=float)
        stats: list[GenerationStat] = []

        for iteration in range(1, self.config.max_iter + 1):
            evaluations = [self._evaluate(candidate) for candidate in population]
            order = self._sort_indices(evaluations)
            population = population[order]
            evaluations = [evaluations[index] for index in order]
            alpha, beta, delta = population[0], population[1], population[2]
            next_population = population.copy()
            next_evaluations: list[CandidateEvaluation] = []
            top_k = max(3, int(np.ceil(self.config.pop_size * self.config.top_k_ratio)))

            for index, (current, current_evaluation) in enumerate(
                zip(population, evaluations, strict=True)
            ):
                gwo_candidate = self._gwo_candidate(current, alpha, beta, delta, iteration, rng)
                gwo_evaluation = self._evaluate(gwo_candidate)
                candidates = [(current, current_evaluation), (gwo_candidate, gwo_evaluation)]
                if self.config.use_alo_local and (
                    index < top_k or rng.random() < self.config.local_prob
                ):
                    alo_candidate = alo_local_exploitation(
                        gwo_candidate,
                        alpha,
                        iteration,
                        self.config.max_iter,
                        self.problem.lower_bounds,
                        self.problem.upper_bounds,
                        rng,
                        self.config,
                    )
                    alo_evaluation = self._evaluate(alo_candidate)
                    candidates.append((alo_candidate, alo_evaluation))
                selected_candidate, selected_evaluation = self._select_best(candidates)
                next_population[index] = selected_candidate
                next_evaluations.append(selected_evaluation)

            iteration_best = next_evaluations[self._sort_indices(next_evaluations)[0]]
            if best is None or self._is_better(iteration_best, best):
                best = iteration_best
            if self.config.elite_preserve and best is not None:
                worst_index = self._sort_indices(next_evaluations)[-1]
                next_population[worst_index] = best.candidate
                next_evaluations[worst_index] = best
            population = next_population
            assert best is not None
            history[iteration - 1] = best.cost
            stats.append(self._generation_stat(iteration, next_evaluations))

        assert best is not None
        raw_best_paths = decode_candidate(best.candidate, self.problem)
        coordinator = ConflictAwareCoordinator(self.scenario, self.scenario.cruise_speed)
        coordination = coordinator.coordinate(raw_best_paths)
        return HGALOResult(
            algorithm="HGALO_PYTHON_COORDINATED_RECONSTRUCTION",
            provenance="Available MATLAB HGALO operators; CA-HGALO MATLAB entrypoint unavailable.",
            seed=self.config.seed,
            best_candidate=best.candidate,
            raw_best_trajectories=raw_best_paths,
            best_trajectories=coordination.repaired_trajectories,
            optimizer_evaluation=best.evaluation,
            coordination=coordination,
            convergence_history=history,
            generation_stats=tuple(stats),
        )

    def _initial_population(self, rng: np.random.Generator) -> FloatArray:
        if self.config.use_chaos_init:
            return chaotic_initialize(self.config.pop_size, self.problem, rng)
        population = rng.uniform(
            self.problem.lower_bounds,
            self.problem.upper_bounds,
            size=(self.config.pop_size, self.problem.dimensions),
        )
        population[0] = self.problem.seed
        return population

    def _gwo_candidate(
        self,
        current: FloatArray,
        alpha: FloatArray,
        beta: FloatArray,
        delta: FloatArray,
        iteration: int,
        rng: np.random.Generator,
    ) -> FloatArray:
        if not self.config.use_gwo_guidance:
            return current.copy()
        return gwo_guided_update(
            current,
            alpha,
            beta,
            delta,
            iteration,
            self.config.max_iter,
            self.problem.lower_bounds,
            self.problem.upper_bounds,
            rng,
        )

    def _evaluate(self, candidate: FloatArray) -> CandidateEvaluation:
        projected = project_candidate(candidate, self.problem)
        if self.config.use_feasibility_repair:
            repaired_candidate, trajectories, _ = repair_candidate(projected, self.problem)
        else:
            repaired_candidate = projected
            trajectories = decode_candidate(projected, self.problem)
        evaluation = self.evaluator.evaluate(trajectories)
        return CandidateEvaluation(
            candidate=repaired_candidate,
            trajectories=trajectories,
            evaluation=evaluation,
            violation=_violation(evaluation),
        )

    def _sort_indices(self, evaluations: list[CandidateEvaluation]) -> list[int]:
        return sorted(range(len(evaluations)), key=lambda index: self._sort_key(evaluations[index]))

    def _sort_key(self, evaluation: CandidateEvaluation) -> tuple[float, float, float]:
        if self.config.use_deb_feasibility_rules:
            return (0.0 if evaluation.success else 1.0, evaluation.violation, evaluation.cost)
        return (evaluation.cost, evaluation.violation, 0.0 if evaluation.success else 1.0)

    def _select_best(
        self, candidates: list[tuple[FloatArray, CandidateEvaluation]]
    ) -> tuple[FloatArray, CandidateEvaluation]:
        best_candidate, best_evaluation = candidates[0]
        for candidate, evaluation in candidates[1:]:
            if self._is_better(evaluation, best_evaluation):
                best_candidate, best_evaluation = candidate, evaluation
        return best_candidate, best_evaluation

    def _is_better(self, first: CandidateEvaluation, second: CandidateEvaluation) -> bool:
        return self._sort_key(first) < self._sort_key(second)

    def _generation_stat(
        self, generation: int, evaluations: list[CandidateEvaluation]
    ) -> GenerationStat:
        feasible = np.asarray([evaluation.success for evaluation in evaluations], dtype=bool)
        threat_failures = np.asarray(
            [
                not evaluation.evaluation.constraint_details.threat_feasible
                for evaluation in evaluations
            ],
            dtype=bool,
        )
        return GenerationStat(
            generation=generation,
            best_cost=min(evaluation.cost for evaluation in evaluations),
            feasible_ratio=float(feasible.mean()),
            best_violation=min(evaluation.violation for evaluation in evaluations),
            threat_failure_ratio=float(threat_failures.mean()),
        )


def _violation(evaluation: EvaluationResult) -> float:
    details = evaluation.constraint_details
    return float(
        int(not details.terrain_feasible)
        + int(not details.threat_feasible)
        + int(not details.turning_feasible)
        + int(not details.spatial_feasible)
        + int(not details.temporal_feasible)
    )
