import torch

from shats.grouping import FeaturesGroupingStrategy
from shats.utils import infer_binary_feature_indices,build_mixed_interpolation_path, integrated_gradients_groups_direct


def test_infer_binary_feature_indices_detects_binary_features():
    dataset = [
        torch.tensor([[0.0, 0.2], [1.0, 0.4]]),
        torch.tensor([[1.0, 0.3], [0.0, 0.5]]),
    ]

    indices = infer_binary_feature_indices(dataset)

    assert indices == [0]


def test_build_mixed_interpolation_path_switches_binary_features_halfway():
    baseline = torch.tensor([[0.0, 0.0]])
    x = torch.tensor([[1.0, 1.0]])

    path = build_mixed_interpolation_path(
        baseline=baseline,
        x=x,
        steps=6,
        categorical_feature_indices=[0],
    )

    assert torch.all(path[:3, :, 0] == 0.0)
    assert torch.all(path[3:, :, 0] == 1.0)
    assert torch.allclose(path[:, 0, 1], torch.linspace(0.0, 1.0, steps=6))


def test_integrated_gradients_groups_direct_handles_binary_and_continuous_features():
    def model_wrapper(batch: torch.Tensor) -> torch.Tensor:
        score = batch.sum(dim=(1, 2))
        prob = torch.sigmoid(score)
        return torch.stack([1.0 - prob, prob], dim=1)

    grouping = FeaturesGroupingStrategy(groups_num=2)
    baseline = torch.tensor([[0.0, 0.0]])
    x = torch.tensor([[1.0, 1.0]])

    attributions = integrated_gradients_groups_direct(
        model_wrapper=model_wrapper,
        x=x,
        baseline=baseline,
        grouping_strategy=grouping,
        class_idx=1,
        steps=20,
        categorical_feature_indices=[0],
    )

    assert attributions.shape == (2,)
    assert torch.all(attributions > 0)