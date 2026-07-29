"""Tests for delayed and lossy inter-UAV communication."""

from __future__ import annotations

import numpy as np

from multiuav.envs.communication import CommunicationChannel, CommunicationConfig


def test_delay_hides_neighbour_until_delivery_step() -> None:
    """A receiver cannot use an in-range packet before its configured arrival step."""
    channel = CommunicationChannel(
        CommunicationConfig(
            enabled=True,
            range=100.0,
            delay_steps=2,
            drop_probability=0.0,
            max_staleness_steps=3,
        ),
        np.random.default_rng(7),
    )
    channel.reset(2)
    channel.broadcast(
        positions=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        velocities=np.zeros((2, 3)),
        active_mask=np.array([True, True]),
        step=0,
    )

    assert not channel.knowledge_for(0, step=1).valid[1]
    channel.deliver(step=2)
    knowledge = channel.knowledge_for(0, step=2)

    assert knowledge.valid[0]
    assert knowledge.valid[1]
    assert knowledge.ages[1] == 2
    np.testing.assert_allclose(knowledge.positions[1], [5.0, 0.0, 0.0])


def test_lossy_channel_discards_packet_without_creating_a_false_neighbor() -> None:
    """A dropped message remains invalid rather than being replaced by current truth."""
    channel = CommunicationChannel(
        CommunicationConfig(
            enabled=True,
            range=100.0,
            delay_steps=0,
            drop_probability=1.0,
            max_staleness_steps=1,
        ),
        np.random.default_rng(11),
    )
    channel.reset(2)
    channel.broadcast(
        positions=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        velocities=np.zeros((2, 3)),
        active_mask=np.array([True, True]),
        step=0,
    )
    channel.deliver(step=0)

    assert not channel.knowledge_for(0, step=0).valid[1]


def test_stale_delivered_neighbor_becomes_invalid() -> None:
    """Delivered data expires at the configured age instead of pretending to be current."""
    channel = CommunicationChannel(
        CommunicationConfig(
            enabled=True,
            range=100.0,
            delay_steps=0,
            drop_probability=0.0,
            max_staleness_steps=1,
        ),
        np.random.default_rng(13),
    )
    channel.reset(2)
    channel.broadcast(
        positions=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        velocities=np.zeros((2, 3)),
        active_mask=np.array([True, True]),
        step=0,
    )
    channel.deliver(step=0)

    assert channel.knowledge_for(0, step=1).valid[1]
    assert not channel.knowledge_for(0, step=2).valid[1]


def test_delivered_neighbor_remains_visible_until_staleness_after_sender_stops() -> None:
    """Sender termination cannot erase a packet that was already delivered to the receiver."""
    channel = CommunicationChannel(
        CommunicationConfig(
            enabled=True,
            range=100.0,
            delay_steps=0,
            drop_probability=0.0,
            max_staleness_steps=1,
        ),
        np.random.default_rng(19),
    )
    channel.reset(2)
    channel.broadcast(
        positions=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        velocities=np.zeros((2, 3)),
        active_mask=np.array([True, True]),
        step=0,
    )
    channel.deliver(step=0)
    channel.broadcast(
        positions=np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
        velocities=np.zeros((2, 3)),
        active_mask=np.array([True, False]),
        step=1,
    )

    knowledge = channel.knowledge_for(0, step=1)

    assert knowledge.valid[1]
    assert knowledge.ages[1] == 1
    np.testing.assert_allclose(knowledge.positions[1], [5.0, 0.0, 0.0])


def test_knowledge_dead_reckons_delivered_packet_and_grows_uncertainty() -> None:
    """A valid stale packet is predicted forward without exposing current neighbour truth."""
    channel = CommunicationChannel(
        CommunicationConfig(
            enabled=True,
            range=100.0,
            delay_steps=0,
            drop_probability=0.0,
            max_staleness_steps=3,
            prediction_dt=0.5,
            uncertainty_growth_per_step=2.0,
        ),
        np.random.default_rng(17),
    )
    channel.reset(2)
    channel.broadcast(
        positions=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        velocities=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        active_mask=np.array([True, True]),
        step=0,
    )
    channel.deliver(step=0)

    knowledge = channel.knowledge_for(0, step=2)

    np.testing.assert_allclose(knowledge.positions[1], [5.0, 0.0, 0.0])
    np.testing.assert_allclose(knowledge.predicted_positions[1], [9.0, 0.0, 0.0])
    assert knowledge.position_uncertainty[1] == 4.0
