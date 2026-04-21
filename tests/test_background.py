import torch

from shats.utils import (
    BackgroundDatasetStrategy,
    infer_background_dataset,
    resolve_background_dataset,
)


def _make_dataset():
    return [
        torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    ]


def test_resolve_background_dataset_with_explicit_dataset_returns_clone():
    background_dataset = _make_dataset()[:2]

    resolved = resolve_background_dataset(background_dataset=background_dataset)

    assert len(resolved) == 2
    assert all(torch.equal(a, b) for a, b in zip(resolved, background_dataset))
    assert resolved[0] is not background_dataset[0]


def test_infer_background_dataset_random_returns_requested_size():
    train_dataset = _make_dataset()

    background_dataset = infer_background_dataset(
        train_dataset=train_dataset,
        background_size=3,
        strategy=BackgroundDatasetStrategy.RANDOM,
        random_state=7,
    )

    assert len(background_dataset) == 3
    assert all(item.shape == train_dataset[0].shape for item in background_dataset)


def test_infer_background_dataset_entropy_selects_high_entropy_samples():
    train_dataset = _make_dataset()

    background_dataset = infer_background_dataset(
        train_dataset=train_dataset,
        background_size=1,
        strategy=BackgroundDatasetStrategy.ENTROPY,
        entropy_bins=2,
    )

    assert torch.equal(background_dataset[0], train_dataset[1])


def test_infer_background_dataset_stratified_preserves_class_proportions():
    train_dataset = _make_dataset() + [
        torch.tensor([[2.0, 2.0], [2.0, 2.0]]),
        torch.tensor([[3.0, 3.0], [3.0, 3.0]]),
        torch.tensor([[4.0, 4.0], [4.0, 4.0]]),
        torch.tensor([[5.0, 5.0], [5.0, 5.0]]),
        torch.tensor([[6.0, 6.0], [6.0, 6.0]]),
    ]
    train_labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

    background_dataset = infer_background_dataset(
        train_dataset=train_dataset,
        background_size=5,
        strategy=BackgroundDatasetStrategy.STRATIFIED,
        train_labels=train_labels,
        random_state=3,
    )

    class_zero_count = sum(
        any(torch.equal(item, train_dataset[index]) for index in range(6))
        for item in background_dataset
    )
    class_one_count = 5 - class_zero_count

    assert class_zero_count == 3
    assert class_one_count == 2


def test_infer_background_dataset_kmeans_returns_centroids_with_expected_shape():
    train_dataset = _make_dataset()

    background_dataset = infer_background_dataset(
        train_dataset=train_dataset,
        background_size=2,
        strategy=BackgroundDatasetStrategy.KMEANS,
        random_state=5,
        kmeans_max_iter=20,
    )

    assert len(background_dataset) == 2
    assert all(item.shape == train_dataset[0].shape for item in background_dataset)