import pytest
import torch

from data.schema import TaskSpec
from model.coordinate_generator import CoordinateGenerator, CoordinateGeneratorConfig
from server.public_initialization import (
    PublicInitializationArtifact,
    PublicTaskObjective,
    learn_public_center,
    load_public_initialization,
    pretrain_coordinate_generator,
    public_instance_description,
    save_public_initialization,
    split_public_examples,
    weighted_prompt_svd,
)


def test_public_examples_split_into_disjoint_round_robin_subsets() -> None:
    rows = tuple(range(8))
    subsets = split_public_examples(rows, num_instances=4, min_examples_per_instance=2)
    assert subsets == ((0, 4), (1, 5), (2, 6), (3, 7))
    assert split_public_examples(rows, num_instances=1, min_examples_per_instance=2) == (rows,)
    with pytest.raises(ValueError, match="not enough public examples"):
        split_public_examples(rows, num_instances=5, min_examples_per_instance=2)


def test_public_instance_description_cycles_then_retags_domain() -> None:
    spec = TaskSpec.from_mapping(
        {
            "task_id": "sst2_sentiment",
            "dataset_path": "dummy",
            "dataset_name": None,
            "task_type": "classification",
            "input_fields": ["sentence"],
            "target_field": "label",
            "metric": "accuracy",
            "description": {
                "op": "sentiment_classification",
                "in": "sentence",
                "out": "categorical_label",
                "dom": "movie_review",
                "lang": "en",
            },
            "description_variants": [
                {
                    "op": "sentiment_classification",
                    "in": "sentence",
                    "out": "categorical_label",
                    "dom": "movie_review",
                    "lang": "en",
                },
                {
                    "op": "review_polarity_classification",
                    "in": "review_text",
                    "out": "categorical_label",
                    "dom": "film_review",
                    "lang": "en",
                },
            ],
        }
    )
    first = public_instance_description(spec, 0)
    second = public_instance_description(spec, 1)
    third = public_instance_description(spec, 2)
    assert first.op == "sentiment_classification"
    assert second.op == "review_polarity_classification"
    assert third.op == first.op
    assert third.domain == "movie_review_view1"
    assert len({first.serialize(), second.serialize(), third.serialize()}) == 3


def test_weighted_public_svd_returns_orthonormal_directions_and_coordinates() -> None:
    center = torch.zeros(2, 3, dtype=torch.float64)
    first = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    second = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    differences = torch.stack((2.0 * first, second, first + second))
    prompts = center + differences.reshape(3, 2, 3)

    basis, coordinates, singular_values = weighted_prompt_svd(
        prompts,
        center,
        torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64),
        num_basis=2,
    )

    matrix = basis.reshape(2, -1).T
    torch.testing.assert_close(
        matrix.T @ matrix,
        torch.eye(2, dtype=torch.float64),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(coordinates @ matrix.T, differences, atol=1.0e-6, rtol=0.0)
    assert singular_values.shape == (3,)


def test_unit_task_svd_equalizes_singular_values_across_task_scales() -> None:
    center = torch.zeros(1, 2)
    large = torch.tensor([[10.0, 0.0]])
    small = torch.tensor([[0.0, 1.0]])
    prompts = torch.stack((center + large, center + small))
    weights = torch.tensor([0.5, 0.5])

    raw_basis, _, raw_s = weighted_prompt_svd(
        prompts, center, weights, num_basis=2, svd_energy="raw"
    )
    unit_basis, unit_coords, unit_s = weighted_prompt_svd(
        prompts, center, weights, num_basis=2, svd_energy="unit_task"
    )

    assert float(raw_s[0] / raw_s[1]) > 5.0
    assert float(unit_s[0] / unit_s[1]) < 1.05
    recon = unit_coords @ unit_basis.reshape(2, -1)
    original = torch.stack((large.reshape(-1), small.reshape(-1)))
    torch.testing.assert_close(recon, original, atol=1.0e-5, rtol=0.0)


def test_label_priority_svd_keeps_class_axis_when_qa_is_larger() -> None:
    center = torch.zeros(1, 2)
    qa = torch.tensor([[12.0, 0.0]])
    nli = torch.tensor([[0.0, 8.0]])
    prompts = torch.stack((center + qa, center + nli))
    weights = torch.tensor([0.5, 0.5])
    types = ("question_answering", "natural_language_inference")

    raw_basis, _, _ = weighted_prompt_svd(
        prompts, center, weights, num_basis=1, svd_energy="raw", task_types=types
    )
    priority_basis, _, _ = weighted_prompt_svd(
        prompts,
        center,
        weights,
        num_basis=1,
        svd_energy="label_priority",
        task_types=types,
    )
    raw_dir = raw_basis.reshape(-1)
    priority_dir = priority_basis.reshape(-1)
    assert abs(float(raw_dir[0])) > abs(float(raw_dir[1]))
    assert abs(float(priority_dir[1])) > abs(float(priority_dir[0]))

    full_basis, full_coords, _ = weighted_prompt_svd(
        prompts,
        center,
        weights,
        num_basis=2,
        svd_energy="label_priority",
        task_types=types,
    )
    recon = full_coords @ full_basis.reshape(2, -1)
    original = torch.stack((qa.reshape(-1), nli.reshape(-1)))
    torch.testing.assert_close(recon, original, atol=1.0e-5, rtol=0.0)


def test_label_priority_svd_requires_task_types() -> None:
    center = torch.zeros(1, 2)
    prompts = torch.stack((center + torch.tensor([[1.0, 0.0]]), center + torch.tensor([[0.0, 1.0]])))
    with pytest.raises(ValueError, match="requires task_types"):
        weighted_prompt_svd(
            prompts,
            center,
            torch.tensor([0.5, 0.5]),
            num_basis=1,
            svd_energy="label_priority",
        )


def test_public_svd_rejects_more_directions_than_effective_rank() -> None:
    center = torch.zeros(2, 2)
    direction = torch.ones(2, 2)
    prompts = torch.stack((center + direction, center + 2.0 * direction))

    with pytest.raises(ValueError, match="insufficient rank"):
        weighted_prompt_svd(
            prompts,
            center,
            torch.tensor([0.5, 0.5]),
            num_basis=2,
        )


def test_public_generator_pretraining_reduces_weighted_coordinate_error() -> None:
    torch.manual_seed(12)
    generator = CoordinateGenerator(
        CoordinateGeneratorConfig(
            embedding_dim=3,
            num_basis=2,
            hidden_dim=8,
            num_layers=1,
        )
    )
    embeddings = torch.eye(3)
    prototype = torch.tensor([1.0, 1.0, 1.0]) / (3.0**0.5)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    weights = torch.tensor([0.2, 0.3, 0.5])

    with torch.no_grad():
        before = generator(embeddings, prototype.expand_as(embeddings))
        before_loss = (weights * (before - targets).square().sum(dim=-1)).sum()
    pretrain_coordinate_generator(
        generator,
        embeddings,
        prototype,
        targets,
        weights,
        steps=200,
        learning_rate=1.0e-2,
    )
    with torch.no_grad():
        after = generator(embeddings, prototype.expand_as(embeddings))
        after_loss = (weights * (after - targets).square().sum(dim=-1)).sum()

    assert float(after_loss) < float(before_loss) * 0.05


def test_public_initialization_artifact_round_trip(tmp_path) -> None:
    generator = CoordinateGenerator(
        CoordinateGeneratorConfig(
            embedding_dim=2,
            num_basis=1,
            hidden_dim=4,
            num_layers=1,
        )
    )
    artifact = PublicInitializationArtifact(
        center=torch.zeros(2, 3),
        basis=torch.nn.functional.normalize(torch.ones(1, 2, 3).reshape(1, -1)).reshape(1, 2, 3),
        task_ids=("task_a", "task_b"),
        task_weights=torch.tensor([0.25, 0.75]),
        task_coordinates=torch.tensor([[1.0], [-1.0]]),
        task_embeddings=torch.eye(2),
        public_prototype=torch.tensor([1.0, 1.0]) / (2.0**0.5),
        singular_values=torch.tensor([2.0, 1.0]),
        generator_state={
            name: value.detach().clone() for name, value in generator.state_dict().items()
        },
    )

    path = save_public_initialization(tmp_path / "public.pt", artifact)
    loaded = load_public_initialization(path)

    assert loaded.task_ids == artifact.task_ids
    torch.testing.assert_close(loaded.center, artifact.center)
    torch.testing.assert_close(loaded.basis, artifact.basis)


def test_public_center_microbatch_backward_matches_joint_graph() -> None:
    torch.manual_seed(0)
    initial = torch.ones(2, 3)

    def quadratic(prompt: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prompt - target).square().mean()

    joint_tasks = (
        PublicTaskObjective(
            task_id="short",
            embedding=torch.ones(2),
            num_examples=1,
            loss=lambda prompt: quadratic(prompt, prompt.new_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])),
        ),
        PublicTaskObjective(
            task_id="long",
            embedding=torch.ones(2),
            num_examples=3,
            loss=lambda prompt: quadratic(prompt, prompt.new_tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])),
        ),
    )
    expected = learn_public_center(joint_tasks, initial, steps=3, learning_rate=0.05)

    backward_calls = {"count": 0}

    def accumulate(prompt: torch.Tensor, scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        backward_calls["count"] += 1
        value = scale * quadratic(prompt, target)
        value.backward()
        return value.detach()

    micro_tasks = (
        PublicTaskObjective(
            task_id="short",
            embedding=torch.ones(2),
            num_examples=1,
            loss=lambda prompt: quadratic(prompt, prompt.new_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])),
            accumulate_backward=lambda prompt, scale: accumulate(
                prompt,
                scale,
                prompt.new_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            ),
        ),
        PublicTaskObjective(
            task_id="long",
            embedding=torch.ones(2),
            num_examples=3,
            loss=lambda prompt: quadratic(prompt, prompt.new_tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])),
            accumulate_backward=lambda prompt, scale: accumulate(
                prompt,
                scale,
                prompt.new_tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]),
            ),
        ),
    )
    actual = learn_public_center(micro_tasks, initial, steps=3, learning_rate=0.05)

    torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=1.0e-5)
    assert backward_calls["count"] == 6

