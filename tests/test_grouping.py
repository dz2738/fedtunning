import torch

from server.grouping import GroupingConfig, SemanticGrouper


def test_cosine_grouping_creates_and_reuses_semantic_groups() -> None:
    grouper = SemanticGrouper(
        GroupingConfig(assignment_threshold=0.8, freeze_during_normal_training=True)
    )
    assignments = grouper.fit(
        {
            "client_a": torch.tensor([1.0, 0.0, 0.0]),
            "client_b": torch.tensor([0.98, 0.10, 0.0]),
            "client_c": torch.tensor([0.0, 1.0, 0.0]),
        }
    )

    assert assignments["client_a"] == assignments["client_b"]
    assert assignments["client_a"] != assignments["client_c"]
    assert grouper.frozen
    torch.testing.assert_close(
        grouper.centroid(assignments["client_a"]).norm(),
        torch.tensor(1.0),
    )


def test_frozen_grouping_does_not_reroute_existing_client() -> None:
    grouper = SemanticGrouper(GroupingConfig(assignment_threshold=0.8))
    assignments = grouper.fit(
        {
            "client_a": torch.tensor([1.0, 0.0]),
            "client_b": torch.tensor([0.0, 1.0]),
        }
    )
    original_group = assignments["client_a"]

    decision = grouper.assign("client_a", torch.tensor([0.0, 1.0]))

    assert decision.group_id == original_group
    assert not decision.created


def test_removed_discrete_grouping_config_is_rejected() -> None:
    try:
        GroupingConfig.from_mapping(
            {
                "field_weights": {"op": 1.0},
                "support_threshold": 0.5,
            }
        )
    except ValueError as error:
        assert "discrete field-support grouping was removed" in str(error)
    else:
        raise AssertionError("legacy discrete grouping configuration was accepted")
