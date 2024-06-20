from itertools import combinations

def all_subsets(subventanas):
    all_subsets_con = []
    all_numbers = list(range(0, subventanas))
    
    # Generar subconjuntos que contienen la feature
    for r in range(0, len(all_numbers) + 2):
        subsets_con = list(combinations(all_numbers, r))
        for subset in subsets_con:
            all_subsets_con.append(sorted(list(subset)))
    
    return all_subsets_con

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
