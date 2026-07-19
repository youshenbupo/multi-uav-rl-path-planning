"""Deterministic delayed and lossy communication for multi-UAV actor knowledge."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class CommunicationConfig:
    """Explicit communication impairment settings for one environment episode."""

    enabled: bool = False
    range: float = float("inf")
    delay_steps: int = 0
    drop_probability: float = 0.0
    max_staleness_steps: int = 0
    prediction_dt: float = 1.0
    uncertainty_growth_per_step: float = 0.0

    def __post_init__(self) -> None:
        if self.range <= 0.0 or np.isnan(self.range):
            raise ValueError("Communication range must be positive and not NaN.")
        if self.delay_steps < 0 or self.max_staleness_steps < 0:
            raise ValueError("Communication delay and max staleness must be nonnegative.")
        if not 0.0 <= self.drop_probability <= 1.0:
            raise ValueError("Communication drop probability must be in [0, 1].")
        if self.prediction_dt <= 0.0 or not np.isfinite(self.prediction_dt):
            raise ValueError("Communication prediction_dt must be finite and positive.")
        if self.uncertainty_growth_per_step < 0.0 or not np.isfinite(
            self.uncertainty_growth_per_step
        ):
            raise ValueError("Communication uncertainty growth must be finite and nonnegative.")


@dataclass(frozen=True)
class AgentKnowledgeState:
    """An actor's true self-state and only the neighbour data it has received."""

    positions: FloatArray
    velocities: FloatArray
    valid: BoolArray
    ages: IntArray
    predicted_positions: FloatArray
    position_uncertainty: FloatArray

    def __post_init__(self) -> None:
        positions = _readonly_float_matrix(self.positions, "positions")
        velocities = _readonly_float_matrix(self.velocities, "velocities")
        valid = np.asarray(self.valid, dtype=bool)
        ages = np.asarray(self.ages, dtype=np.int64)
        predicted_positions = _readonly_float_matrix(
            self.predicted_positions, "predicted_positions"
        )
        position_uncertainty = np.asarray(self.position_uncertainty, dtype=float)
        if positions.shape != velocities.shape or positions.shape[1:] != (3,):
            raise ValueError("Knowledge positions and velocities must have shape [N, 3].")
        if valid.shape != (len(positions),) or ages.shape != (len(positions),):
            raise ValueError("Knowledge validity and ages must have shape [N].")
        if predicted_positions.shape != positions.shape:
            raise ValueError("Knowledge predicted_positions must have shape [N, 3].")
        if position_uncertainty.shape != (len(positions),) or not np.isfinite(
            position_uncertainty
        ).all():
            raise ValueError("Knowledge position_uncertainty must be finite with shape [N].")
        if np.any(position_uncertainty < 0.0):
            raise ValueError("Knowledge position_uncertainty must be nonnegative.")
        if np.any(ages < -1):
            raise ValueError("Knowledge ages must be at least -1.")
        valid = valid.copy()
        ages = ages.copy()
        valid.setflags(write=False)
        ages.setflags(write=False)
        position_uncertainty = position_uncertainty.copy()
        position_uncertainty.setflags(write=False)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "ages", ages)
        object.__setattr__(self, "predicted_positions", predicted_positions)
        object.__setattr__(self, "position_uncertainty", position_uncertainty)


@dataclass(frozen=True)
class _QueuedPacket:
    """One immutable directed broadcast awaiting its deterministic delivery step."""

    receiver: int
    sender: int
    source_step: int
    delivery_step: int
    position: FloatArray
    velocity: FloatArray


class CommunicationChannel:
    """Own packet delivery and expose communication-limited per-agent knowledge."""

    def __init__(self, config: CommunicationConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self._count = 0
        self._packets: list[_QueuedPacket] = []
        self._received_positions = np.empty((0, 0, 3), dtype=float)
        self._received_velocities = np.empty((0, 0, 3), dtype=float)
        self._received_steps = np.empty((0, 0), dtype=np.int64)
        self._current_positions: FloatArray | None = None
        self._current_velocities: FloatArray | None = None
        self._current_active: BoolArray | None = None

    def reset(self, count: int) -> None:
        """Clear packets and received state for a fixed number of UAVs."""
        if count < 1:
            raise ValueError("Communication channel needs at least one UAV.")
        self._count = count
        self._packets = []
        self._received_positions = np.zeros((count, count, 3), dtype=float)
        self._received_velocities = np.zeros((count, count, 3), dtype=float)
        self._received_steps = np.full((count, count), -1, dtype=np.int64)
        self._current_positions = None
        self._current_velocities = None
        self._current_active = None

    def broadcast(
        self,
        *,
        positions: FloatArray,
        velocities: FloatArray,
        active_mask: BoolArray,
        step: int,
    ) -> None:
        """Enqueue one directed packet for each eligible active sender/receiver pair."""
        self._require_reset()
        position_values, velocity_values, active_values = self._validate_broadcast(
            positions, velocities, active_mask, step
        )
        self._current_positions = position_values.copy()
        self._current_velocities = velocity_values.copy()
        self._current_active = active_values.copy()
        if not self.config.enabled:
            self._receive_all_current(step)
            return
        for sender in range(self._count):
            if not active_values[sender]:
                continue
            for receiver in range(self._count):
                if receiver == sender or not active_values[receiver]:
                    continue
                distance = np.linalg.norm(position_values[sender] - position_values[receiver])
                if distance > self.config.range:
                    continue
                if self.rng.random() < self.config.drop_probability:
                    continue
                self._packets.append(
                    _QueuedPacket(
                        receiver=receiver,
                        sender=sender,
                        source_step=step,
                        delivery_step=step + self.config.delay_steps,
                        position=position_values[sender].copy(),
                        velocity=velocity_values[sender].copy(),
                    )
                )

    def deliver(self, *, step: int) -> None:
        """Deliver every due packet in deterministic source and receiver order."""
        self._require_reset()
        if step < 0:
            raise ValueError("Communication delivery step must be nonnegative.")
        due = [packet for packet in self._packets if packet.delivery_step <= step]
        self._packets = [packet for packet in self._packets if packet.delivery_step > step]
        ordered_packets = sorted(
            due,
            key=lambda item: (item.delivery_step, item.source_step, item.receiver, item.sender),
        )
        for packet in ordered_packets:
            if packet.source_step < self._received_steps[packet.receiver, packet.sender]:
                continue
            self._received_positions[packet.receiver, packet.sender] = packet.position
            self._received_velocities[packet.receiver, packet.sender] = packet.velocity
            self._received_steps[packet.receiver, packet.sender] = packet.source_step

    def knowledge_for(self, receiver: int, *, step: int) -> AgentKnowledgeState:
        """Return current self truth plus fresh received neighbour values only."""
        self._require_current_state()
        if receiver < 0 or receiver >= self._count:
            raise ValueError("Communication receiver index is out of range.")
        if step < 0:
            raise ValueError("Communication knowledge step must be nonnegative.")
        assert self._current_positions is not None
        assert self._current_velocities is not None
        assert self._current_active is not None
        positions = np.zeros((self._count, 3), dtype=float)
        velocities = np.zeros((self._count, 3), dtype=float)
        valid = np.zeros(self._count, dtype=bool)
        ages = np.full(self._count, -1, dtype=np.int64)
        predicted_positions = np.zeros((self._count, 3), dtype=float)
        position_uncertainty = np.zeros(self._count, dtype=float)
        if self._current_active[receiver]:
            positions[receiver] = self._current_positions[receiver]
            velocities[receiver] = self._current_velocities[receiver]
            predicted_positions[receiver] = self._current_positions[receiver]
            valid[receiver] = True
            ages[receiver] = 0
        for sender in range(self._count):
            if sender == receiver or not self._current_active[sender]:
                continue
            source_step = int(self._received_steps[receiver, sender])
            age = step - source_step
            if source_step < 0 or age > self.config.max_staleness_steps:
                continue
            positions[sender] = self._received_positions[receiver, sender]
            velocities[sender] = self._received_velocities[receiver, sender]
            predicted_positions[sender] = (
                positions[sender] + age * self.config.prediction_dt * velocities[sender]
            )
            position_uncertainty[sender] = self.config.uncertainty_growth_per_step * age
            valid[sender] = True
            ages[sender] = age
        return AgentKnowledgeState(
            positions,
            velocities,
            valid,
            ages,
            predicted_positions,
            position_uncertainty,
        )

    def _receive_all_current(self, step: int) -> None:
        assert self._current_positions is not None
        assert self._current_velocities is not None
        assert self._current_active is not None
        for receiver in range(self._count):
            for sender in range(self._count):
                if receiver == sender or not self._current_active[sender]:
                    continue
                self._received_positions[receiver, sender] = self._current_positions[sender]
                self._received_velocities[receiver, sender] = self._current_velocities[sender]
                self._received_steps[receiver, sender] = step

    def _validate_broadcast(
        self, positions: FloatArray, velocities: FloatArray, active_mask: BoolArray, step: int
    ) -> tuple[FloatArray, FloatArray, BoolArray]:
        if step < 0:
            raise ValueError("Communication broadcast step must be nonnegative.")
        position_values = np.asarray(positions, dtype=float)
        velocity_values = np.asarray(velocities, dtype=float)
        active_values = np.asarray(active_mask, dtype=bool)
        if position_values.shape != (self._count, 3) or velocity_values.shape != (self._count, 3):
            raise ValueError("Communication positions and velocities must have shape [N, 3].")
        if active_values.shape != (self._count,):
            raise ValueError("Communication active mask must have shape [N].")
        if not np.isfinite(position_values).all() or not np.isfinite(velocity_values).all():
            raise ValueError("Communication broadcast values must be finite.")
        return position_values, velocity_values, active_values

    def _require_reset(self) -> None:
        if self._count < 1:
            raise RuntimeError("Communication channel must be reset before use.")

    def _require_current_state(self) -> None:
        self._require_reset()
        if self._current_positions is None:
            raise RuntimeError(
                "Communication channel must receive a broadcast before knowledge access."
            )


def _readonly_float_matrix(values: FloatArray, name: str) -> FloatArray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"Knowledge {name} must be a finite matrix.")
    copied = matrix.copy()
    copied.setflags(write=False)
    return copied
