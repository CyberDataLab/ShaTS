"""
This module contains utility functions for generating subsets for the estimation
"""
from __future__ import annotations

import math
import random
from enum import Enum
from typing import NamedTuple, Sequence, Callable

import numpy as np
import torch
from torch import Tensor

from .grouping import (
    AbstractGroupingStrategy,
    FeaturesGroupingStrategy,
    MultifeaturesGroupingStrategy,
    TimeGroupingStrategy,
)

class StrategySubsets(Enum):
    """
    Strategy used to generate subsets for Shapley estimation.
    """

    EXACT = 1
    APPROX = 2

class BackgroundDatasetStrategy(str, Enum):
    """
    Strategy used to build a background dataset from the training dataset.
    """

    RANDOM = "random"
    ENTROPY = "entropy"
    STRATIFIED = "stratified"
    KMEANS = "kmeans"

Subset = tuple[int, ...]
Predictors2subsetsDict = dict[tuple[int, int], tuple[list[Subset], list[Subset]]]


class GeneratedSubsets(NamedTuple):
    """
    Named tuple representing the generated subsets.
    Attributes:
        predictors_to_subsets: A dictionary mapping (predictor, size) to subsets.
        all_subsets: A list of all unique subsets generated.
    """
    predictors_to_subsets: Predictors2subsetsDict
    all_subsets: list[Subset]



def validate_window_dataset(dataset: Sequence[Tensor], dataset_name: str) -> None:
    """
    Validate that a dataset is a non-empty collection of 2D tensors with the same shape.
    """
    if not dataset:
        raise ValueError(f"{dataset_name} must contain at least one tensor.")

    first_shape = tuple(dataset[0].shape)
    if len(first_shape) != 2:
        raise ValueError(f"Each tensor in {dataset_name} must have shape [time, features].")

    for index, item in enumerate(dataset):
        if not isinstance(item, Tensor):
            raise TypeError(f"All items in {dataset_name} must be torch.Tensor instances.")
        if tuple(item.shape) != first_shape:
            raise ValueError(
                f"All tensors in {dataset_name} must have the same shape. "
                f"Found {tuple(item.shape)} at position {index}, expected {first_shape}."
            )


def clone_dataset(dataset: Sequence[Tensor]) -> list[Tensor]:
    """
    Return a detached clone of a tensor dataset.
    """
    return [item.detach().clone() for item in dataset]


def resolve_background_dataset(
    background_dataset: Sequence[Tensor] | None = None,
    train_dataset: Sequence[Tensor] | None = None,
    background_size: int | None = None,
    background_dataset_strategy: BackgroundDatasetStrategy = BackgroundDatasetStrategy.RANDOM,
    train_labels: Sequence[int] | Tensor | None = None,
    random_state: int | None = None,
    entropy_bins: int = 32,
    kmeans_max_iter: int = 100,
    kmeans_tolerance: float = 1e-4,
) -> list[Tensor]:
    """
    Resolve the background dataset from an explicit dataset or infer it from the training dataset.
    """
    if background_dataset is not None and train_dataset is not None:
        raise ValueError("Provide either background_dataset or train_dataset, but not both.")

    if background_dataset is not None:
        validate_window_dataset(background_dataset, "background_dataset")
        return clone_dataset(background_dataset)

    if train_dataset is None:
        raise ValueError("Either background_dataset or train_dataset must be provided.")

    if background_size is None:
        raise ValueError("background_size must be provided when train_dataset is used.")

    return infer_background_dataset(
        train_dataset=train_dataset,
        background_size=background_size,
        strategy=background_dataset_strategy,
        train_labels=train_labels,
        random_state=random_state,
        entropy_bins=entropy_bins,
        kmeans_max_iter=kmeans_max_iter,
        kmeans_tolerance=kmeans_tolerance,
    )


def infer_background_dataset(
    train_dataset: Sequence[Tensor],
    background_size: int,
    strategy: BackgroundDatasetStrategy = BackgroundDatasetStrategy.RANDOM,
    train_labels: Sequence[int] | Tensor | None = None,
    random_state: int | None = None,
    entropy_bins: int = 32,
    kmeans_max_iter: int = 100,
    kmeans_tolerance: float = 1e-4,
) -> list[Tensor]:
    """
    Infer a background dataset from the training dataset.
    """
    validate_window_dataset(train_dataset, "train_dataset")

    if background_size < 1:
        raise ValueError("background_size must be greater than zero.")

    if strategy != BackgroundDatasetStrategy.KMEANS and background_size > len(train_dataset):
        raise ValueError(
            "background_size cannot be greater than the number of training instances "
            "for the selected strategy."
        )

    if strategy == BackgroundDatasetStrategy.RANDOM:
        return _infer_background_random(train_dataset, background_size, random_state)
    if strategy == BackgroundDatasetStrategy.ENTROPY:
        return _infer_background_entropy(train_dataset, background_size, entropy_bins)
    if strategy == BackgroundDatasetStrategy.STRATIFIED:
        return _infer_background_stratified(
            train_dataset, background_size, train_labels, random_state
        )
    if strategy == BackgroundDatasetStrategy.KMEANS:
        return _infer_background_kmeans(
            train_dataset,
            background_size,
            random_state=random_state,
            max_iter=kmeans_max_iter,
            tolerance=kmeans_tolerance,
        )

    raise ValueError(f"Unsupported background dataset strategy: {strategy!r}.")


def infer_binary_feature_indices(
    dataset: Sequence[Tensor],
    round_decimals: int | None = 8,
) -> list[int]:
    """
    Infer the indices of binary features from a window dataset.
    """
    validate_window_dataset(dataset, "dataset")

    stacked = torch.stack(clone_dataset(dataset), dim=0)
    _, _, num_features = stacked.shape

    if round_decimals is not None and torch.is_floating_point(stacked):
        factor = float(10**round_decimals)
        stacked = torch.round(stacked * factor) / factor

    binary_feature_indices: list[int] = []
    for feature_idx in range(num_features):
        unique_values = torch.unique(stacked[:, :, feature_idx])
        if unique_values.numel() == 2:
            binary_feature_indices.append(feature_idx)

    return binary_feature_indices


def generate_subsets(
    total_groups: int,
    total_wanted_subsets: int,
    strategy: StrategySubsets = StrategySubsets.APPROX,
) -> GeneratedSubsets:
    """
    Generate subsets for a given number of groups and a specified strategy.

    Args:
        nGroups (int): Number of groups.
        nSubsets (int): Number of subsets to generate for each group and size.
        strategy (StrategySubsets): Strategy for subset generation. Options are:
            - StrategySubsets.EXACT: Generate all possible subsets of each size for each group.
            - StrategySubsets.APPROX: Generate `nSubsets` subsets for each size per group.

    Returns:
        Tuple[Dict[Tuple[int, int], Tuple[List[List[int]], List[List[int]]]], List[List[int]]]:
            - A dictionary where keys are tuples of (predictor, size), and values are tuples of:
              - A list of subsets containing the predictor.
              - A list of subsets excluding the predictor.
            - A flattened list of all unique subsets generated.

    Raises:
        ValueError: If the number of groups is less than 1 or the number of subsets is negative.
    """
    if total_groups < 1:
        raise ValueError("nGroups must be at least 1.")
    if total_wanted_subsets < 0:
        raise ValueError("nSubsets must be non-negative.")

    all_subsets: list[set[Subset]] = [set() for _ in range(total_groups + 1)]
    subset_dict = {}

    for group in range(total_groups):
        for size in range(total_groups):
            num_of_subsets_to_generate = _calculate_num_subsets_to_generate(
                total_groups, total_wanted_subsets, size, strategy
            )

            # Generate subsets
            subsets_without_group = [
                subset for subset in all_subsets[size] if group not in subset
            ]
            subsets_with_group: list[tuple[int, ...]] = [
                tuple(sorted(subset + (group,))) for subset in subsets_without_group
            ]

            remaining_nums = list(range(total_groups))
            remaining_nums.remove(group)

            # Avoid duplicates by maintaining intersections
            intersection = list[Subset]()

            for i, subset in enumerate(subsets_without_group):
                if subsets_with_group[i] in all_subsets[size + 1]:
                    intersection.append(subset)

            subsets_without_group = sorted(
                subsets_without_group, key=lambda x: x in intersection, reverse=False
            )
            subsets_with_group = sorted(
                subsets_with_group, key=lambda x: x in intersection, reverse=False
            )

            while len(subsets_without_group) < num_of_subsets_to_generate:
                random_subset_without = tuple(
                    sorted(random.sample(remaining_nums, size))
                )
                random_subset_with = tuple(sorted(random_subset_without + (group,)))

                if random_subset_without not in all_subsets[size]:
                    all_subsets[size].add(random_subset_without)
                    subsets_without_group.append(random_subset_without)
                    subsets_with_group.append(random_subset_with)

                if random_subset_with not in all_subsets[size + 1]:
                    all_subsets[size + 1].add(random_subset_with)

            subsets_with_group = subsets_with_group[:num_of_subsets_to_generate]

            for subset in subsets_with_group:
                if subset not in all_subsets[size + 1]:
                    all_subsets[size + 1].add(subset)

            subsets_without_group = subsets_without_group[:num_of_subsets_to_generate]
            subset_dict[(group, size)] = (
                [list(subset) for subset in subsets_with_group],
                [list(subset) for subset in subsets_without_group],
            )

    # Flatten all subsets
    flatenned_subsets = [
        tuple(subset) for sizeSubsets in all_subsets for subset in sizeSubsets
    ]

    return GeneratedSubsets(subset_dict, flatenned_subsets)



def estimate_m(total_features: int, total_desired_subsets: int) -> int:
    """
    Estimate the internal subset parameter m from the desired number of subsets.
    """
    limit = (
        2
        * sum((index + 1) ** (2 / 3) for index in range(total_features))
        / total_features ** (2 / 3)
    )
    limit = round(limit)

    if total_desired_subsets <= limit:
        return limit

    step = max((limit**2 - limit) // 20, 1)
    values = range(limit, limit**2, step)
    list_values = list(values)

    subset_counts: list[int] = []
    for value in list_values:
        _, subsets_total = generate_subsets(total_features, value)
        subset_counts.append(len(subsets_total))

    x = np.array(list_values)
    y = np.array(subset_counts)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sum((x - mean_x) ** 2)
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x

    m = (total_desired_subsets - intercept) / slope

    if np.isinf(m) or np.isnan(m) or m < 0:
        return limit

    return round(m)


def _calculate_num_subsets_to_generate(
    total_groups: int,
    total_wanted_subsets: int,
    size: int,
    strategy: StrategySubsets,
) -> int:
    num_subsets = math.floor(
        total_wanted_subsets
        * (size + 1) ** (2 / 3)
        / sum((index + 1) ** (2 / 3) for index in range(total_groups))
    )
    num_subsets = min(num_subsets, math.comb(total_groups - 1, size))

    if strategy == StrategySubsets.EXACT:
        num_subsets = math.comb(total_groups - 1, size)

    return max(num_subsets, 1)


def _infer_background_random(
    train_dataset: Sequence[Tensor],
    background_size: int,
    random_state: int | None,
) -> list[Tensor]:
    rng = random.Random(random_state)
    selected_indices = rng.sample(range(len(train_dataset)), background_size)
    return [train_dataset[index].detach().clone() for index in selected_indices]


def _infer_background_entropy(
    train_dataset: Sequence[Tensor],
    background_size: int,
    entropy_bins: int,
) -> list[Tensor]:
    entropies = []
    for index, sample in enumerate(train_dataset):
        flattened = sample.detach().cpu().reshape(-1).numpy()
        bins = min(max(2, entropy_bins), max(2, flattened.size))
        hist, _ = np.histogram(flattened, bins=bins)
        probs = hist[hist > 0].astype(np.float64)
        probs = probs / probs.sum()
        entropy = float(-(probs * np.log2(probs)).sum())
        entropies.append((index, entropy))

    selected_indices = [index for index, _ in sorted(entropies, key=lambda item: item[1], reverse=True)[:background_size]]
    return [train_dataset[index].detach().clone() for index in selected_indices]


def _infer_background_stratified(
    train_dataset: Sequence[Tensor],
    background_size: int,
    train_labels: Sequence[int] | Tensor | None,
    random_state: int | None,
) -> list[Tensor]:
    if train_labels is None:
        raise ValueError("train_labels must be provided when using the STRATIFIED strategy.")

    labels_tensor = torch.as_tensor(train_labels, dtype=torch.long)
    if labels_tensor.numel() != len(train_dataset):
        raise ValueError("train_labels must have the same length as train_dataset.")

    rng = random.Random(random_state)

    unique_labels, counts = torch.unique(labels_tensor, return_counts=True)
    total = int(counts.sum().item())

    raw_allocations = counts.float() / total * background_size
    allocations = torch.floor(raw_allocations).to(torch.long)

    remainder = background_size - int(allocations.sum().item())
    if remainder > 0:
        fractional = raw_allocations - allocations.float()
        order = torch.argsort(fractional, descending=True)
        for idx in order[:remainder]:
            allocations[idx] += 1

    selected_indices: list[int] = []
    for label, allocation in zip(unique_labels.tolist(), allocations.tolist()):
        label_indices = torch.where(labels_tensor == label)[0].tolist()
        if allocation > len(label_indices):
            raise ValueError(
                "Not enough samples to satisfy the requested stratified background size."
            )
        if allocation > 0:
            selected_indices.extend(rng.sample(label_indices, allocation))

    return [train_dataset[index].detach().clone() for index in selected_indices]


def _infer_background_kmeans(
    train_dataset: Sequence[Tensor],
    background_size: int,
    random_state: int | None,
    max_iter: int,
    tolerance: float,
) -> list[Tensor]:
    if background_size > len(train_dataset):
        raise ValueError(
            "background_size cannot be greater than the number of training instances "
            "when using the KMEANS strategy."
        )

    stacked = torch.stack([sample.detach().reshape(-1).float() for sample in train_dataset], dim=0)
    generator = torch.Generator(device=stacked.device)
    if random_state is not None:
        generator.manual_seed(random_state)

    initial_indices = torch.randperm(stacked.shape[0], generator=generator)[:background_size]
    centroids = stacked[initial_indices].clone()

    for _ in range(max_iter):
        distances = torch.cdist(stacked, centroids)
        assignments = torch.argmin(distances, dim=1)

        new_centroids = []
        for cluster_idx in range(background_size):
            cluster_points = stacked[assignments == cluster_idx]
            if cluster_points.numel() == 0:
                replacement_index = torch.randint(
                    low=0,
                    high=stacked.shape[0],
                    size=(1,),
                    generator=generator,
                ).item()
                new_centroids.append(stacked[replacement_index])
            else:
                new_centroids.append(cluster_points.mean(dim=0))

        updated_centroids = torch.stack(new_centroids, dim=0)
        shift = torch.norm(updated_centroids - centroids)
        centroids = updated_centroids

        if float(shift.item()) <= tolerance:
            break

    template_shape = train_dataset[0].shape
    template_dtype = train_dataset[0].dtype if torch.is_floating_point(train_dataset[0]) else torch.float32

    return [
        centroid.reshape(template_shape).to(dtype=template_dtype).detach().clone()
        for centroid in centroids
    ]

## Utilities for IG implementation


def build_group_directions(
    x: Tensor,
    baseline: Tensor,
    grouping_strategy: AbstractGroupingStrategy,
) -> Tensor:
    """
    Build one direction tensor per ShaTS group.
    """
    diff = x - baseline
    time_steps, num_features = diff.shape
    groups_num = grouping_strategy.groups_num

    directions = torch.zeros(
        groups_num,
        time_steps,
        num_features,
        device=x.device,
        dtype=x.dtype,
    )

    if isinstance(grouping_strategy, TimeGroupingStrategy):
        for group_idx in range(groups_num):
            directions[group_idx, group_idx, :] = diff[group_idx, :]

    elif isinstance(grouping_strategy, FeaturesGroupingStrategy):
        for group_idx in range(groups_num):
            directions[group_idx, :, group_idx] = diff[:, group_idx]

    elif isinstance(grouping_strategy, MultifeaturesGroupingStrategy):
        for group_idx, feature_indices in enumerate(grouping_strategy.custom_groups):
            directions[group_idx, :, feature_indices] = diff[:, feature_indices]

    else:
        for group_idx in range(groups_num):
            directions[group_idx] = diff

    return directions


def build_mixed_interpolation_path(
    baseline: Tensor,
    x: Tensor,
    steps: int,
    categorical_feature_indices: Sequence[int] | None = None,
) -> Tensor:
    """
    Build a path from baseline to x using linear interpolation for continuous features and a two-stage path for binary features.
    """
    if steps < 2:
        raise ValueError("steps must be at least 2.")

    baseline = baseline.detach()
    x = x.detach()

    alphas = torch.linspace(0.0, 1.0, steps=steps, device=x.device, dtype=x.dtype).view(-1, 1, 1)
    path = baseline.unsqueeze(0) + alphas * (x - baseline).unsqueeze(0)

    if categorical_feature_indices:
        switch_index = steps // 2
        categorical_feature_indices = list(dict.fromkeys(categorical_feature_indices))

        path[:switch_index, :, categorical_feature_indices] = baseline[:, categorical_feature_indices]
        path[switch_index:, :, categorical_feature_indices] = x[:, categorical_feature_indices]

    return path


def aggregate_attributions_by_group(
    attributions: Tensor,
    grouping_strategy: AbstractGroupingStrategy,
) -> Tensor:
    """
    Aggregate input-space attributions at the ShaTS group level.
    """
    groups_num = grouping_strategy.groups_num
    aggregated = torch.zeros(groups_num, device=attributions.device, dtype=attributions.dtype)

    if isinstance(grouping_strategy, TimeGroupingStrategy):
        for group_idx in range(groups_num):
            aggregated[group_idx] = attributions[group_idx, :].sum()

    elif isinstance(grouping_strategy, FeaturesGroupingStrategy):
        for group_idx in range(groups_num):
            aggregated[group_idx] = attributions[:, group_idx].sum()

    elif isinstance(grouping_strategy, MultifeaturesGroupingStrategy):
        for group_idx, feature_indices in enumerate(grouping_strategy.custom_groups):
            aggregated[group_idx] = attributions[:, feature_indices].sum()

    else:
        total = attributions.sum()
        aggregated[:] = total / max(groups_num, 1)

    return aggregated


def integrated_gradients_groups_direct(
    model_wrapper: Callable[[Tensor], Tensor],
    x: Tensor,
    baseline: Tensor,
    grouping_strategy: AbstractGroupingStrategy,
    class_idx: int = 0,
    steps: int = 50,
    device: torch.device | str = "cpu",
    categorical_feature_indices: Sequence[int] | None = None,
) -> Tensor:
    """
    Compute group-level Integrated Gradients from baseline to x.
    """
    device = torch.device(device)
    x = x.to(device)
    baseline = baseline.to(device)

    path = build_mixed_interpolation_path(
        baseline=baseline,
        x=x,
        steps=steps,
        categorical_feature_indices=categorical_feature_indices,
    )
    path.requires_grad_(True)

    outputs = model_wrapper(path)
    if outputs.dim() != 2:
        raise ValueError("model_wrapper must return a tensor of shape [batch, nclass].")

    target = outputs[:, class_idx].sum()
    grads = torch.autograd.grad(target, path)[0]

    segment_deltas = path[1:] - path[:-1]
    segment_grads = 0.5 * (grads[1:] + grads[:-1])
    input_attributions = (segment_grads * segment_deltas).sum(dim=0)

    return aggregate_attributions_by_group(
        attributions=input_attributions,
        grouping_strategy=grouping_strategy,
    )