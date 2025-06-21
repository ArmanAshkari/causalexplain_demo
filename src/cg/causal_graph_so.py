import networkx as nx
import pickle
from .profile_generation import generate_profiles

import pygraphviz as pgv


def create_causal_graph():
    """
    """
    # Geneate Causal Graph
    V = ['Age', 'Gender', 'Country', 'RaceEthnicity', 'FormalEducation', 'UndergradMajor', 'DevType', 'HoursComputer', 'Dependents', 'ConvertedSalary']
    target = 'ConvertedSalary'
    g = nx.DiGraph()
    g.add_nodes_from(V)

    g.add_edge('Age', 'FormalEducation')
    g.add_edge('Age', 'DevType')
    g.add_edge('Age', 'Dependents')
    g.add_edge('Age', 'ConvertedSalary')

    g.add_edge('Gender', 'FormalEducation')
    g.add_edge('Gender', 'UndergradMajor')
    g.add_edge('Gender', 'DevType')
    g.add_edge('Gender', 'ConvertedSalary')

    g.add_edge('Country', 'RaceEthnicity')
    g.add_edge('Country', 'FormalEducation')
    g.add_edge('Country', 'ConvertedSalary')
    
    g.add_edge('FormalEducation', 'DevType')
    g.add_edge('FormalEducation', 'UndergradMajor')

    g.add_edge('RaceEthnicity', 'ConvertedSalary')
    
    g.add_edge('UndergradMajor', 'DevType')

    g.add_edge('DevType', 'HoursComputer')
    g.add_edge('DevType', 'ConvertedSalary')
    
    g.add_edge('Dependents', 'HoursComputer')
    
    g.add_edge('HoursComputer', 'ConvertedSalary')

    return g, V, target


def generate_profile_from_causal_graph(g, V, target):
    """
    """
    profiles = generate_profiles(G=g, V=V, target=target, lim_disj=2)
    profiles = [__ for _ in profiles for __ in _]

    with open('data/so/profiles_dis2.pkl', 'wb') as f:
        pickle.dump(profiles, f)


def load_profiles_from_disk():
    """
    """
    with open('data/so/profiles_dis2.pkl', 'rb') as f:
        profiles = pickle.load(f)

    return profiles


def test():
    """
    """
    profiles = load_profiles_from_disk()
    print(len(profiles))

if __name__ == '__main__':
    """
    python -m src.cg.causal_graph_so
    """
    # # Create causal graph
    # g, V, target = create_causal_graph()

    # # # Generate profiles
    # generate_profile_from_causal_graph(g, V, target)

    # Load profiles
    profiles = load_profiles_from_disk()
    print(f'Loaded {len(profiles)} profiles from disk.')

    print(type(profiles))
    print(profiles[1000])
    print(type(profiles[0]))
    print(profiles[1000][0])  # Should be a tuple
    print(type(profiles[0][0]))
    print(profiles[1000][0][0])  # Should be a string

    for profile in profiles:
        print(profile)