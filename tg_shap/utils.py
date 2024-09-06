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


def generateRandomSubsets(n, m):
    
    todos_conjuntos = [set() for _ in range(n+1)]
    dict_pairs = {}
    
    for predictor in range(n):
        for size in range(n):
            
            m_actual = min(m, math.comb(n - 1, size))
            
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
            
            while len(candidatos_sin) < m_actual:
                subset_sin = tuple(sorted(random.sample(all_numbers, size)))
                subset_con = tuple(sorted(subset_sin + (predictor,)))
                
                if subset_sin not in todos_conjuntos[size]:
                    todos_conjuntos[size].add(subset_sin)
                    candidatos_sin.append(subset_sin)
                    candidatos_con.append(subset_con)        
                
                if subset_con not in todos_conjuntos[size+1]:
                    todos_conjuntos[size+1].add(subset_con)
                    
            candidatos_con = candidatos_con[:m_actual]
            
            for conjunto in candidatos_con:
                if conjunto not in todos_conjuntos[size+1]:
                    todos_conjuntos[size+1].add(conjunto)
            
            candidatos_sin = candidatos_sin[:m_actual]
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

