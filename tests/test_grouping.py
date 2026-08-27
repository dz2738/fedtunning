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


def test_agglomerative_grouping_is_stable_and_uses_task_items_once() -> None:
    config = GroupingConfig(
        strategy="agglomerative",
        assignment_threshold=0.9,
    )
    embeddings = {
        "task_b": torch.tensor([0.0, 1.0]),
        "task_a2": torch.tensor([0.98, 0.08]),
        "task_a1": torch.tensor([1.0, 0.0]),
    }

    first = SemanticGrouper(config)
    second = SemanticGrouper(config)
    first_assignments = first.fit(embeddings)
    second_assignments = second.fit(dict(reversed(tuple(embeddings.items()))))

    assert first_assignments == second_assignments
    assert first_assignments["task_a1"] == first_assignments["task_a2"]
    assert first_assignments["task_a1"] != first_assignments["task_b"]


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
