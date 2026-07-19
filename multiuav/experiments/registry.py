"""Explicit unified baseline availability registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodDefinition:
    """One benchmark method and whether this repository can execute it."""

    name: str
    availability: str
    reason: str


class MethodRegistry:
    """Keep executable methods and unavailable external baselines equally auditable."""

    def __init__(self) -> None:
        legacy_without_dynamic_adapter = (
            "straight_line_greedy",
            "python_ca_hgalo",
            "rule_conflict_coordinator",
            "mappo",
            "predictive_graph_mappo",
            "hierarchical_graph_mappo",
            "bc_initialized_hierarchical_graph_mappo",
        )
        unavailable = ("astar", "rrt_star", "orca")
        self._methods = {
            "full_method": MethodDefinition(
                "full_method", "available", "hierarchical checkpoint dynamic-world adapter"
            )
        } | {
            name: MethodDefinition(
                name,
                "unavailable",
                "legacy implementation exists but no unified dynamic-world adapter is implemented",
            )
            for name in legacy_without_dynamic_adapter
        } | {
            name: MethodDefinition(name, "unavailable", "not implemented in this repository")
            for name in unavailable
        }

    def resolve(self, name: str) -> MethodDefinition:
        """Return a structured record or reject unknown labels before an experiment starts."""
        try:
            return self._methods[name]
        except KeyError as error:
            raise ValueError(f"Unknown benchmark method: {name}") from error
