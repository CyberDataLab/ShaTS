from itertools import combinations
import math
import random
from enum import Enum
from typing import Dict, List, Set, Tuple


class StrategySubsets(Enum):
    """Enum representing the strategy for generating subsets."""
    EXACT = 1
    APPROX_MK = 2

class StrategyGrouping(Enum):
    """Enum representing the strategy for grouping."""
    TIME = 1
    FEATURE = 2
    PROCESS = 3

class StrategyPrediction(Enum):
    ONECLASS = 1
    MULTICLASS = 2


## FUNCION ACTUALIZADA
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
                #print("Exacto con iguall!!")
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



######################### LEGACY CODE #########################



#deprecated
def all_subsets(subventanas):
    print("He hecho este cambio 4 veces")
    all_subsets_con = []
    all_numbers = list(range(0, subventanas))
    
    # Generar subconjuntos que contienen la feature
    for r in range(0, len(all_numbers) + 2):
        subsets_con = list(combinations(all_numbers, r))
        for subset in subsets_con:
            all_subsets_con.append(sorted(list(subset)))
    
    return all_subsets_con
#deprecated
def all_subsets_without(subventanas, feature):
    all_subsets_con = []
    all_subsets_sin = []
    all_numbers = list(range(0, subventanas))
    all_numbers.remove(feature)
    
    # Generar subconjuntos que contienen la feature
    for r in range(1, len(all_numbers) + 2):
        subsets_con = list(combinations(all_numbers, r))
        for subset in subsets_con:
            all_subsets_con.append(sorted(list(subset) + [feature]))
    
    # Generar subconjuntos que no contienen la feature
    for r in range(0, len(all_numbers) + 1):
        subsets_sin = list(combinations(all_numbers, r))
        for subset in subsets_sin:
            all_subsets_sin.append(sorted(list(subset)))
    
    return all_subsets_con, all_subsets_sin

#DEPRECATED
def generateRandomSubsets(n, m):
    print("He hecho este cambio 8 veces")
    
    todos_conjuntos = [set() for _ in range(n+1)]
    dict_pairs = {}
    
    for predictor in range(n):
        for size in range(n):
            
            #m_actual = min(m, math.comb(n - 1, size))
            m_k = math.floor(m * (size +1)**(2/3) / sum([(k+1)**(2/3) for k in range(0, n)]))
            m_k = min(m_k, math.comb(n - 1, size))

            if m_k == 0:
                m_k = 1
            
            m_k = math.comb(n - 1, size)
            
            candidatos_sin = [conjunto for conjunto in todos_conjuntos[size] if predictor not in conjunto]
            candidatos_con = [(tuple(sorted(conjunto + (predictor,)))) for conjunto in candidatos_sin]
            
            all_numbers = list(range(0, n))
            all_numbers.remove(predictor)
                        
            suma = 0
            interseccion = []
            for i in range(len(candidatos_sin)):
                if candidatos_con[i] in todos_conjuntos[size+1]:
                    suma +=1
                    interseccion.append(candidatos_sin[i])
            
            candidatos_sin = sorted(candidatos_sin, key=lambda x: x in interseccion, reverse=False)
            candidatos_con = sorted(candidatos_con, key=lambda x: x in interseccion, reverse=False)
            
            while len(candidatos_sin) < m_k:
                subset_sin = tuple(sorted(random.sample(all_numbers, size)))
                subset_con = tuple(sorted(subset_sin + (predictor,)))
                
                if subset_sin not in todos_conjuntos[size]:
                    todos_conjuntos[size].add(subset_sin)
                    candidatos_sin.append(subset_sin)
                    candidatos_con.append(subset_con)        
                
                if subset_con not in todos_conjuntos[size+1]:
                    todos_conjuntos[size+1].add(subset_con)
                    
            candidatos_con = candidatos_con[:m_k]
            
            for conjunto in candidatos_con:
                if conjunto not in todos_conjuntos[size+1]:
                    todos_conjuntos[size+1].add(conjunto)
            
            candidatos_sin = candidatos_sin[:m_k]
            candidatos_sin = [list(conjunto) for conjunto in candidatos_sin]
            candidatos_con = [list(conjunto) for conjunto in candidatos_con]
            dict_pairs[(predictor, size)] = (candidatos_con, candidatos_sin)
    
    #aplanar todos_conjuntos
    todos_conjuntos = [list(conjuntos) for conjuntos in todos_conjuntos]
    # hacer solo un conjunto de todos los conjuntos
    todos_conjuntos = [conjunto for conjuntos in todos_conjuntos for conjunto in conjuntos]
    #hacer una lista de conjuntos
    todos_conjuntos = [list(conjunto) for conjunto in todos_conjuntos]

    return dict_pairs, todos_conjuntos


## n = Numero de grupos
## m = Numero de subconjuntos de cada tamaño para cada grupo
## mode = 'auto' o 'exact': 'auto' para generar m subconjuntos de cada tamaño para cada grupo, 'exact' para generar todos los subconjuntos posibles






