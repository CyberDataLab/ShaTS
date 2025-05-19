"""
This module contains utility functions for generating subsets for the estimation
"""
import math
import random
from enum import Enum
from typing import NamedTuple
import itertools
import numpy as np

class StrategySubsets(Enum):
    """Enum representing the subsets generation strategy."""

    EXACT = 1
    APPROX = 2

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


def _calculate_num_subsets_to_generate(
    total_groups: int, total_wanted_subsets: int, size: int, strategy: StrategySubsets
):
    num = math.floor(
        total_wanted_subsets
        * (size + 1) ** (2 / 3)
        / sum([(k + 1) ** (2 / 3) for k in range(total_groups)])
    )
    num = min(num, math.comb(total_groups - 1, size))

    if strategy.value == StrategySubsets.EXACT.value:
        num = math.comb(total_groups - 1, size)

    if num == 0:
        num = 1

    return num


def estimate_m(total_features: int, total_desired_subsets: int) -> int:
    limit = (
        2
        * sum((i + 1) ** (2 / 3) for i in range(total_features))
        / total_features ** (2 / 3)
    )
    limit = round(limit)

    if total_desired_subsets <= limit:
        return limit

    step = max((limit**2 - limit) // 20, 1)
    values = range(limit, limit**2, step)
    list_values = list(values)

    sizes = list[int]()

    for value in list_values:
        _, subsets_total = generate_subsets(total_features, value)
        sizes.append(len(subsets_total))

    x = np.array(list_values)
    y = np.array(sizes)

    # Calculate regression coefficients
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Calculate slope (m) and intersection (b)
    numer = np.sum((x - mean_x) * (y - mean_y))
    denom = np.sum((x - mean_x) ** 2)
    slope = numer / denom
    intercept = mean_y - slope * mean_x

    # Calculate final `m`
    m = (total_desired_subsets - intercept) / slope

    if np.isinf(m) or np.isnan(m):
        return limit

    if m < 0:
        return limit

    return round(m)


def generate_subsets_alternative(num_groups: int, num_subsets: int, max_size: int):
    """
    Generate subsets of size up to max_size containing a specific group.
    Additionally, generate subsets for sizes (M, M-1, ..., M-max_size) with M = num_groups.
 
    Args:
        num_groups (int): Number of groups.
        num_subsets (int): Number of subsets to generate for each group and size.
        max_size (int): Maximum subset size for generation.
 
    Returns:
        Tuple:
            - A dictionary where keys are (group, size) for size in {1, ..., max_size} and (M, M-1, ..., M-max_size),
              and values are lists of subsets containing the group.
            - A flattened list of all unique subsets generated, without repetitions.
    """
    subset_dict = {}
    flattened_subsets = set()
    flattened_subsets.add(tuple())
    #If max_size is greater than the number of groups, we can generate all combinations  
    if max_size > num_groups // 2:
        all_subsets = [list(subset) for i in range(1, num_groups + 1) for subset in itertools.combinations(range(num_groups), i)]
        
        # Filling the dictionary with all possible combinations
        for group in range(num_groups):
            for subset in all_subsets:
                if group in subset:
                    key = (group, len(subset))
                    if key not in subset_dict:
                        subset_dict[key] = []
                    subset_dict[key].append(subset)
        
        return subset_dict, all_subsets
 
    for group in range(num_groups):
        remaining_numbers = [g for g in range(num_groups) if g != group]
 
        # Subsets of sizes 1 to max_size
        for size in range(1, max_size + 1):
            limited_subsets = set()
            attempts = 0
            max_attempts = 10 * num_subsets
 
            while len(limited_subsets) < num_subsets and attempts < max_attempts:
                subset = random.sample(remaining_numbers, size - 1)
                subset.append(group)
                limited_subsets.add(tuple(sorted(subset)))
                attempts += 1
 
            subset_dict[(group, size)] = [list(subset) for subset in limited_subsets]
            flattened_subsets.update(limited_subsets)
 
        # Subsets of sizes M to M- max_size
        M = num_groups
        for size in range(M, M - max_size - 1, -1):
            limited_subsets = set()
            attempts = 0
            max_attempts = 10 * num_subsets
 
            while len(limited_subsets) < num_subsets and attempts < max_attempts:
                subset = random.sample(remaining_numbers, size - 1)
                subset.append(group)
                limited_subsets.add(tuple(sorted(subset)))
                attempts += 1
 
            subset_dict[(group, size)] = [list(subset) for subset in limited_subsets]
            flattened_subsets.update(limited_subsets)
 
    # Ensure all groups are represented in flattened_subsets without duplicates
    unique_flattened_subsets = {tuple(subset) for subset in flattened_subsets}
    return subset_dict, [list(subset) for subset in unique_flattened_subsets]