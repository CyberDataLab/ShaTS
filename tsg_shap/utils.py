import math
import random
from enum import Enum
from typing import Dict, List, Tuple


class StrategySubsets(Enum):
    """Enum representing the strategy for generating subsets."""
    EXACT = 1
    APPROX_MK = 2

class StrategyGrouping(Enum):
    """Enum representing the strategy for grouping."""
    TIME = 1
    FEATURE = 2
    MULTIFEATURE = 3

class StrategyPrediction(Enum):
    ONECLASS = 1
    MULTICLASS = 2


def generate_subsets(num_groups: int, num_subsets: int, strategy: StrategySubsets = StrategySubsets.APPROX_MK) -> Tuple[Dict[Tuple[int, int], Tuple[List[List[int]], List[List[int]]]], List[List[int]]]:
    """
    Generate subsets for a given number of groups and a specified strategy.

    Args:
        num_groups (int): Number of groups.
        num_subsets (int): Number of subsets to generate for each group and size.
        strategy (StrategySubsets): Strategy for subset generation. Options are:
            - StrategySubsets.EXACT: Generate all possible subsets of each size for each group.
            - StrategySubsets.APPROX_MK: Generate approximately `num_subsets` subsets for each size per group.

    Returns:
        Tuple[Dict[Tuple[int, int], Tuple[List[List[int]], List[List[int]]]], List[List[int]]]:
            - A dictionary where keys are tuples of (predictor, size), and values are tuples of:
              - A list of subsets containing the predictor.
              - A list of subsets excluding the predictor.
            - A flattened list of all unique subsets generated.

    Raises:
        ValueError: If the number of groups is less than 1 or the number of subsets is negative.
    """
    if num_groups < 1:
        raise ValueError("num_groups must be at least 1.")
    if num_subsets < 0:
        raise ValueError("num_subsets must be non-negative.")

    all_subsets = [set() for _ in range(num_groups + 1)]
    subset_dict = {}

    for group in range(num_groups):
        for size in range(num_groups):
            # Calculate the number of subsets to generate
            subsets_to_generate = math.floor(num_subsets * (size + 1)**(2/3) / 
                                             sum([(k + 1)**(2/3) for k in range(num_groups)]))
            subsets_to_generate = min(subsets_to_generate, math.comb(num_groups - 1, size))

            if strategy.value == StrategySubsets.EXACT.value:
                subsets_to_generate = math.comb(num_groups - 1, size)

            if subsets_to_generate == 0:
                subsets_to_generate = 1

            # Generate subsets
            subsets_without_group = [subset for subset in all_subsets[size] if group not in subset]
            subsets_with_group = [tuple(sorted(subset + (group,))) for subset in subsets_without_group]

            remaining_numbers = list(range(num_groups))
            remaining_numbers.remove(group)

            # Avoid duplicates by maintaining intersections
            intersection = []
            for i, subset in enumerate(subsets_without_group):
                if subsets_with_group[i] in all_subsets[size + 1]:
                    intersection.append(subset)

            subsets_without_group = sorted(
                subsets_without_group,
                key=lambda x: x in intersection,
                reverse=False
            )
            subsets_with_group = sorted(
                subsets_with_group,
                key=lambda x: x in intersection,
                reverse=False
            )

            while len(subsets_without_group) < subsets_to_generate:
                random_subset_without = tuple(sorted(random.sample(remaining_numbers, size)))
                random_subset_with = tuple(sorted(random_subset_without + (group,)))

                if random_subset_without not in all_subsets[size]:
                    all_subsets[size].add(random_subset_without)
                    subsets_without_group.append(random_subset_without)
                    subsets_with_group.append(random_subset_with)

                if random_subset_with not in all_subsets[size + 1]:
                    all_subsets[size + 1].add(random_subset_with)

            subsets_with_group = subsets_with_group[:subsets_to_generate]

            for subset in subsets_with_group:
                if subset not in all_subsets[size + 1]:
                    all_subsets[size + 1].add(subset)

            subsets_without_group = subsets_without_group[:subsets_to_generate]
            subset_dict[(group, size)] = (
                [list(subset) for subset in subsets_with_group],
                [list(subset) for subset in subsets_without_group]
            )

    # Flatten all subsets
    flattened_subsets = [list(subset) for size_subsets in all_subsets for subset in size_subsets]

    return subset_dict, flattened_subsets

