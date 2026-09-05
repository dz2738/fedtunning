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


def test_agglomerative_stops_at_target_num_groups() -> None:
    embeddings = {
        "sst": torch.tensor([1.0, 0.05, 0.0, 0.0]),
        "ag": torch.tensor([0.98, 0.08, 0.0, 0.0]),
        "boolq": torch.tensor([0.0, 0.0, 1.0, 0.05]),
        "squad": torch.tensor([0.0, 0.0, 0.97, 0.10]),
        "rte": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "xsum": torch.tensor([0.0, 0.0, 0.0, 1.0]),
    }
    grouper = SemanticGrouper(
        GroupingConfig(
            strategy="agglomerative",
            assignment_threshold=0.90,
            target_num_groups=4,
        )
    )
    assignments = grouper.fit(embeddings)

    assert len(set(assignments.values())) == 4
    assert assignments["sst"] == assignments["ag"]
    assert assignments["boolq"] == assignments["squad"]
    assert assignments["rte"] != assignments["xsum"]
    assert assignments["sst"] != assignments["rte"]


def test_agglomerative_does_not_merge_below_threshold() -> None:
    embeddings = {
        "a": torch.tensor([1.0, 0.0, 0.0]),
        "b": torch.tensor([0.0, 1.0, 0.0]),
        "c": torch.tensor([0.0, 0.0, 1.0]),
    }
    grouper = SemanticGrouper(
        GroupingConfig(
            strategy="agglomerative",
            assignment_threshold=0.95,
            target_num_groups=1,
        )
    )
    assignments = grouper.fit(embeddings)

    assert len(set(assignments.values())) == 3


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


def test_near_duplicate_client_descriptions_still_share_a_group() -> None:
    embeddings = {
        "sst/client_000": torch.tensor([1.0, 0.05, 0.0, 0.0]),
        "sst/client_001": torch.tensor([0.99, 0.08, 0.0, 0.0]),
        "ag/client_000": torch.tensor([0.98, 0.09, 0.0, 0.0]),
        "ag/client_001": torch.tensor([0.97, 0.12, 0.0, 0.0]),
        "boolq/client_000": torch.tensor([0.0, 0.0, 1.0, 0.05]),
        "boolq/client_001": torch.tensor([0.0, 0.0, 0.99, 0.08]),
        "squad/client_000": torch.tensor([0.0, 0.0, 0.97, 0.10]),
        "squad/client_001": torch.tensor([0.0, 0.0, 0.96, 0.12]),
        "rte/client_000": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "rte/client_001": torch.tensor([0.02, 0.99, 0.0, 0.0]),
        "xsum/client_000": torch.tensor([0.0, 0.0, 0.0, 1.0]),
        "xsum/client_001": torch.tensor([0.0, 0.0, 0.02, 0.99]),
    }
    grouper = SemanticGrouper(
        GroupingConfig(
            strategy="agglomerative",
            assignment_threshold=0.90,
            target_num_groups=4,
        )
    )
    assignments = grouper.fit(embeddings)

    assert len(set(assignments.values())) == 4
    assert assignments["sst/client_000"] == assignments["sst/client_001"]
    assert assignments["sst/client_000"] == assignments["ag/client_000"]
    assert assignments["boolq/client_000"] == assignments["squad/client_000"]
    assert assignments["rte/client_000"] != assignments["xsum/client_000"]


def test_select_one_client_per_task_returns_hashable_ids() -> None:
    from trainer.holdout import select_one_client_per_task

    class _Client:
        def __init__(self, task_id: str) -> None:
            self.state = type("State", (), {"task_id": task_id})()

    clients = {
        "glue_rte/client_000": _Client("glue_rte"),
        "glue_rte/client_001": _Client("glue_rte"),
        "sst2_sentiment/client_000": _Client("sst2_sentiment"),
        "sst2_sentiment/client_001": _Client("sst2_sentiment"),
    }
    holdout = select_one_client_per_task(clients, seed=42)
    assert len(holdout) == 2
    assert all(isinstance(client_id, str) for client_id in holdout)
    assert set(holdout) <= set(clients)
    assert {client_id.split("/")[0] for client_id in holdout} == {
        "glue_rte",
        "sst2_sentiment",
    }


def test_holdout_cosine_accepts_tensors_on_different_devices() -> None:
    from trainer.holdout import _cosine

    left = torch.tensor([1.0, 0.0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    right = torch.tensor([1.0, 0.0], device=device)

    assert _cosine(left, right) == 1.0
