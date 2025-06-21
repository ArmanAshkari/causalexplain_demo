import copy
import pickle
import numpy as np
import networkx as nx
import math


def DFS(G, s, visited):
    """
    """
    visited.add(s)
    for u in G[s]:
        if u not in visited:
            DFS(G, u, visited)
    return visited


def compute_reachability(G, V, f2i):
    """
    """
    n = len(V)
    reachability_matrix = np.zeros((n, n), dtype=np.int8)

    for v in V:
        visited = set()
        visited = DFS(G, v, visited)

        v2i = f2i[v]
        for w in visited:
            w2i = f2i[w]
            reachability_matrix[v2i, w2i] = 1 
    
    return reachability_matrix


def compute_bag_of_conjunctions(features, f2i, reachability_matrix):
    """
    """
    bag_of_conjunctions = [[(v,) for v in features]] # list(list(tuple))

    while True:
        last_level_conjunctions = bag_of_conjunctions[-1] # list(tuple)
        next_level_conjunctions = []

        for conjunction in last_level_conjunctions: # tuple
            for v in features:
                if v in conjunction:
                    continue

                v2i = f2i[v]
                for j, c in enumerate(conjunction):
                    c2i = f2i[c]

                    if j == 0 and reachability_matrix[v2i][c2i]:
                        new_conjunction = [v]
                        new_conjunction.extend(conjunction)
                        new_conjunction = tuple(new_conjunction)
                        if new_conjunction not in next_level_conjunctions:
                            next_level_conjunctions.append(new_conjunction)
                        break
                    else:
                        _c2i = f2i[conjunction[j-1]]
                        if reachability_matrix[_c2i][v2i] and reachability_matrix[v2i][c2i]:
                            new_conjunction = []
                            new_conjunction.extend(conjunction[0:j])
                            new_conjunction.append(v)
                            new_conjunction.extend(conjunction[j:])
                            new_conjunction = tuple(new_conjunction)
                            if new_conjunction not in next_level_conjunctions:
                                next_level_conjunctions.append(new_conjunction)
                            break
        
        if len(next_level_conjunctions) == 0:
            break
        else:
            bag_of_conjunctions.append(next_level_conjunctions)
    
    return bag_of_conjunctions


def is_collapsible(A, B, f2i, reachability_matrix):
    """
    """
    m, n = len(A), len(B)
    i, j = 0, 0

    while True:
        if i == m:
            _a2i = f2i[A[i-1]]
            if reachability_matrix[_a2i][b2i]:
                return True
            else:
                return False
        
        if j == n:
            return True

        a2i = f2i[A[i]]
        b2i = f2i[B[j]]

        if i == 0: 
            if reachability_matrix[b2i][a2i]:
                j += 1
            else:    
                i += 1
        else:
            _a2i = f2i[A[i-1]]
            if reachability_matrix[_a2i][b2i] and reachability_matrix[b2i][a2i]:
                j += 1
            else:
                i += 1


def compute_bag_of_disjunctions(bag_of_conjunctions, f2i, reachability_matrix, lim):
    """
    """
    all_conjunctions = [conjunction for conjunctions in bag_of_conjunctions for conjunction in conjunctions] # list(tuple)
    bag_of_disjunctions = [[(conjunction,) for conjunction in all_conjunctions]] # list(list(tuple(tuple)))

    while len(bag_of_disjunctions) < lim:        
        last_level_disjunctions = bag_of_disjunctions[-1] # list(tuple(tuple))
        next_level_disjunctions = []

        for disjunction in last_level_disjunctions: # tuple(tuple)
            for conjunction in all_conjunctions: # tuple
                
                if conjunction in disjunction: 
                    continue
            
                flag = False
                for term in disjunction: # tuple
                    if is_collapsible(term, conjunction, f2i, reachability_matrix):
                        flag = True
                        break
                
                if flag:
                    continue

                new_disjunction = list(copy.deepcopy(disjunction)) # tuple(tuple) -> list(tuple)
                new_disjunction.append(conjunction)
                new_disjunction.sort()
                new_disjunction = tuple(new_disjunction) # list(tuple) -> tuple(tuple)
                if new_disjunction not in next_level_disjunctions:
                    next_level_disjunctions.append(new_disjunction) # list(tuple(tuple))

        if len(next_level_disjunctions) == 0:
            break
        else:
            bag_of_disjunctions.append(next_level_disjunctions) # list(list(tuple(tuple)))
        
    return bag_of_disjunctions


def generate_profiles(G, V, target, lim_disj=math.inf):
    """
    """
    features = copy.deepcopy(V)
    features.remove(target)

    f2i = dict(map(reversed, enumerate(V)))

    rm = compute_reachability(G=G, V=V, f2i=f2i)
    bofc = compute_bag_of_conjunctions(features=features, f2i=f2i, reachability_matrix=rm)
    bofd = compute_bag_of_disjunctions(bag_of_conjunctions=bofc, f2i=f2i, reachability_matrix=rm, lim=lim_disj)

    return bofd 