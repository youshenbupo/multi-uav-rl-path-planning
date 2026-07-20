"""Paper-facing protocol validation before long-running multi-seed training starts."""

from pathlib import Path

from multiuav.experiments.core_protocol import load_core_protocol


def test_core_protocol_requires_five_seeds_four_independent_arms_and_shared_cbf() -> None:
    protocol = load_core_protocol(
        Path(__file__).parents[1] / "configs/experiments/core_3uav_mainline.yaml"
    )

    assert len(protocol.seeds) == 5
    assert protocol.initial_num_uavs == 3
    assert protocol.later_num_uavs == (5, 8)
    assert tuple(method.name for method in protocol.methods) == (
        "mlp_mappo",
        "raw_graph_mappo",
        "predictive_graph_mappo",
        "uncertainty_predictive_graph_mappo",
    )
    assert len({method.artifact_prefix for method in protocol.methods}) == 4
