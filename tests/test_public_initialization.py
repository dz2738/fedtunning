import pytest
import torch

from model.coordinate_generator import CoordinateGenerator, CoordinateGeneratorConfig
from server.public_initialization import (
    PublicInitializationArtifact,
    PublicTaskObjective,
    learn_public_center,
    load_public_initialization,
    pretrain_coordinate_generator,
    save_public_initialization,
    weighted_prompt_svd,
)


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

