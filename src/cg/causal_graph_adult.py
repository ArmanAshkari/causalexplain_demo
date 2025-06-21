import networkx as nx
import pickle
from .profile_generation import generate_profiles

import pygraphviz as pgv

def create_causal_graph():
    """
    """
    # Geneate Causal Graph
    # graph taken from https://par.nsf.gov/servlets/purl/10126315
    V = ['age', 'race', 'gender', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'hours-per-week', 'native-country', 'income']
    target = 'income'
    g = nx.DiGraph()
    g.add_nodes_from(V)

    # # Simplified graph
    g.add_edge('age', 'marital-status')
    g.add_edge('native-country', 'marital-status')
    g.add_edge('race', 'marital-status')
    g.add_edge('gender', 'marital-status')
    
    g.add_edge('marital-status', 'education')

    g.add_edge('education', 'hours-per-week')
    g.add_edge('education', 'occupation')
    g.add_edge('education', 'workclass')
    g.add_edge('education', 'relationship')

    g.add_edge('hours-per-week', 'income')
    g.add_edge('workclass', 'income')
    g.add_edge('relationship', 'income')
    g.add_edge('occupation', 'income')


    # Full graph
    # g.add_edge('age', 'marital-status')
    # g.add_edge('native-country', 'marital-status')
    # g.add_edge('race', 'marital-status')
    # g.add_edge('gender', 'marital-status')
    
    # g.add_edge('age', 'education')
    # g.add_edge('native-country', 'education')
    # g.add_edge('race', 'education')
    # g.add_edge('gender', 'education')
    # g.add_edge('marital-status', 'education')

    # g.add_edge('age', 'hours-per-week')
    # g.add_edge('native-country', 'hours-per-week')
    # g.add_edge('race', 'hours-per-week')
    # g.add_edge('gender', 'hours-per-week')
    # g.add_edge('education', 'hours-per-week')
    # g.add_edge('marital-status', 'hours-per-week')

    # g.add_edge('age', 'occupation')
    # g.add_edge('marital-status', 'occupation')
    # g.add_edge('race', 'occupation')
    # g.add_edge('gender', 'occupation')
    # g.add_edge('education', 'occupation')

    # g.add_edge('age', 'workclass')
    # g.add_edge('native-country', 'workclass')
    # g.add_edge('marital-status', 'workclass')
    # g.add_edge('education', 'workclass')

    # g.add_edge('age', 'relationship')
    # g.add_edge('native-country', 'relationship')
    # g.add_edge('gender', 'relationship')
    # g.add_edge('education', 'relationship')
    # g.add_edge('marital-status', 'relationship')

    # g.add_edge('age', 'income')
    # g.add_edge('native-country', 'income')
    # g.add_edge('race', 'income')
    # g.add_edge('gender', 'income')
    # g.add_edge('education', 'income')
    # g.add_edge('marital-status', 'income')
    # g.add_edge('hours-per-week', 'income')
    # g.add_edge('workclass', 'income')
    # g.add_edge('relationship', 'income')
    # g.add_edge('occupation', 'income')

    return g, V, target


def generate_profile_from_causal_graph(g, V, target):
    """
    """
    profiles = generate_profiles(G=g, V=V, target=target, lim_disj=2)
    profiles = [__ for _ in profiles for __ in _]

    with open('data/adult/profiles_dis2.pkl', 'wb') as f:
        pickle.dump(profiles, f)


def load_profiles_from_disk():
    """
    """
    with open('data/adult/profiles_dis2.pkl', 'rb') as f:
        profiles = pickle.load(f)

    return profiles


if __name__ == '__main__':
    """
    python -m src.cg.causal_graph_adult
    """
    # # Create causal graph
    # g, V, target = create_causal_graph()

    # # # Generate profiles
    # generate_profile_from_causal_graph(g, V, target)

    # # Load profiles
    profiles = load_profiles_from_disk()
    print(f'Loaded {len(profiles)} profiles from disk.')

    # print(type(profiles))
    # print(profiles[3000])
    # print(type(profiles[0]))
    # print(profiles[3000][0])  # Should be a tuple
    # print(type(profiles[0][0]))
    # print(profiles[3000][0][0])  # Should be a string

    for profile in profiles:
        print(profile)