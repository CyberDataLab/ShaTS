import math
import random
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple


class StrategySubsets(Enum):
    """Enum representing the strategy for generating subsets."""
    EXACT = 1
    APPROX = 2

class StrategyGrouping(Enum):
    """Enum representing the strategy for grouping."""
    TIME = 1
    FEATURE = 2
    MULTIFEATURE = 3

class StrategyPrediction(Enum):
    ONECLASS = 1
    MULTICLASS = 2


def generateSubsets(nGroups: int, nSubsets: int, strategy: StrategySubsets = StrategySubsets.APPROX) -> Tuple[Dict[Tuple[int, int], Tuple[List[List[int]], List[List[int]]]], List[List[int]]]:
    """
    Generate subsets for a given number of groups and a specified strategy.

    Args:
        nGroups (int): Number of groups.
        nSubsets (int): Number of subsets to generate for each group and size.
        strategy (StrategySubsets): Strategy for subset generation. Options are:
            - StrategySubsets.EXACT: Generate all possible subsets of each size for each group.
            - StrategySubsets.APPROX: Generate approximately `nSubsets` subsets for each size per group.

    Returns:
        Tuple[Dict[Tuple[int, int], Tuple[List[List[int]], List[List[int]]]], List[List[int]]]:
            - A dictionary where keys are tuples of (predictor, size), and values are tuples of:
              - A list of subsets containing the predictor.
              - A list of subsets excluding the predictor.
            - A flattened list of all unique subsets generated.

    Raises:
        ValueError: If the number of groups is less than 1 or the number of subsets is negative.
    """
    if nGroups < 1:
        raise ValueError("nGroups must be at least 1.")
    if nSubsets < 0:
        raise ValueError("nSubsets must be non-negative.")

    allSubsets = [set() for _ in range(nGroups + 1)]
    subsetDict = {}

    for group in range(nGroups):
        for size in range(nGroups):
            # Calculate the number of subsets to generate
            nSubsetsToGenerate = math.floor(nSubsets * (size + 1)**(2/3) / 
                                             sum([(k + 1)**(2/3) for k in range(nGroups)]))
            nSubsetsToGenerate = min(nSubsetsToGenerate, math.comb(nGroups - 1, size))

            if strategy.value == StrategySubsets.EXACT.value:
                nSubsetsToGenerate = math.comb(nGroups - 1, size)

            if nSubsetsToGenerate == 0:
                nSubsetsToGenerate = 1

            # Generate subsets
            subsetsWithoutGroup = [subset for subset in allSubsets[size] if group not in subset]
            subsetsWithGroup = [tuple(sorted(subset + (group,))) for subset in subsetsWithoutGroup]

            remainingNumbers = list(range(nGroups))
            remainingNumbers.remove(group)

            # Avoid duplicates by maintaining intersections
            intersection = []
            for i, subset in enumerate(subsetsWithoutGroup):
                if subsetsWithGroup[i] in allSubsets[size + 1]:
                    intersection.append(subset)

            subsetsWithoutGroup = sorted(
                subsetsWithoutGroup,
                key=lambda x: x in intersection,
                reverse=False
            )
            subsetsWithGroup = sorted(
                subsetsWithGroup,
                key=lambda x: x in intersection,
                reverse=False
            )

            while len(subsetsWithoutGroup) < nSubsetsToGenerate:
                randomSubsetWithout = tuple(sorted(random.sample(remainingNumbers, size)))
                randomSubsetWith = tuple(sorted(randomSubsetWithout + (group,)))

                if randomSubsetWithout not in allSubsets[size]:
                    allSubsets[size].add(randomSubsetWithout)
                    subsetsWithoutGroup.append(randomSubsetWithout)
                    subsetsWithGroup.append(randomSubsetWith)

                if randomSubsetWith not in allSubsets[size + 1]:
                    allSubsets[size + 1].add(randomSubsetWith)

            subsetsWithGroup = subsetsWithGroup[:nSubsetsToGenerate]

            for subset in subsetsWithGroup:
                if subset not in allSubsets[size + 1]:
                    allSubsets[size + 1].add(subset)

            subsetsWithoutGroup = subsetsWithoutGroup[:nSubsetsToGenerate]
            subsetDict[(group, size)] = (
                [list(subset) for subset in subsetsWithGroup],
                [list(subset) for subset in subsetsWithoutGroup]
            )

    # Flatten all subsets
    flatennedSubsets = [list(subset) for sizeSubsets in allSubsets for subset in sizeSubsets]

    return subsetDict, flatennedSubsets


def estimate_m(numFeatures, numSubsetsDesired):

    critico = 2*sum([(i+1)**(2/3) for i in range(numFeatures)])/ numFeatures**(2/3)
    critico = round(critico)

    if numSubsetsDesired <= critico:
        return critico

    step = max((critico**2 - critico) // 20, 1)
    values = range(critico, critico**2, step)
    list_values = list(values)

    sizes = []

    for value in list_values:
        subsets_dict, subsets_total = generateSubsets(numFeatures, value)
        sizes.append(len(subsets_total))

    # Convertir listas a arrays para facilitar cálculos
    X = np.array(list_values)
    y = np.array(sizes)

    # Calcular los coeficientes de la regresión lineal
    n = len(X)
    mean_x = np.mean(X)
    mean_y = np.mean(y)

    # Calcular la pendiente (m) y la intersección (b) de la recta
    numer = np.sum((X - mean_x) * (y - mean_y))
    denom = np.sum((X - mean_x) ** 2)
    slope = numer / denom
    intercept = mean_y - slope * mean_x

    # Calcular el valor de m
    m = (numSubsetsDesired - intercept) / slope

    if np.isinf(m) or np.isnan(m):
        return critico

    if m < 0:
        return critico

    return round(m)

