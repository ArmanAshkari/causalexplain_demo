import networkx as nx
import pickle
from .profile_generation import generate_profiles

import pygraphviz as pgv

def create_causal_graph():
    """
    """
    # Geneate Causal Graph
    # graph taken from https://par.nsf.gov/servlets/purl/10126315
    V = ['is_recid', 'priors_count', 'juv_other_count', 'juv_misd_count', 'juv_fel_count', 'race', 'age_cat', 'sex','score_text']
    target = 'is_recid'
    g = nx.DiGraph()
    g.add_nodes_from(V)

    edges = [
        ('sex', 'juv_fel_count'),
        ('sex', 'juv_other_count'),
        ('sex', 'juv_misd_count'),
        ('sex', 'priors_count'),
        ('sex', 'score_text'),
        ('sex', 'is_recid'),

        ('race', 'juv_fel_count'),
        ('race', 'juv_misd_count'),
        ('race', 'juv_other_count'),
        ('race', 'priors_count'),
        ('race', 'score_text'),
        ('race', 'is_recid'),

        ('age_cat', 'juv_fel_count'),
        ('age_cat', 'juv_misd_count'),
        ('age_cat', 'juv_other_count'),
        ('age_cat', 'priors_count'),
        ('age_cat', 'score_text'),
        ('age_cat', 'is_recid'),

        ('juv_fel_count', 'priors_count'),
        ('juv_fel_count', 'is_recid'),
        ('juv_fel_count', 'score_text'),

        ('juv_misd_count', 'priors_count'),
        ('juv_misd_count', 'is_recid'),
        ('juv_misd_count', 'score_text'),

        ('juv_other_count', 'priors_count'),
        ('juv_other_count', 'is_recid'),
        ('juv_other_count', 'score_text'),
        ('priors_count', 'score_text'),

        ('score_text', 'is_recid'),
    ]

    g.add_edges_from(edges)

    return g, V, target


def generate_profile_from_causal_graph(g, V, target):
    """
    """
    profiles = generate_profiles(G=g, V=V, target=target, lim_disj=2)
    profiles = [__ for _ in profiles for __ in _]

    with open('data/compas/profiles_dis2.pkl', 'wb') as f:
        pickle.dump(profiles, f)


def load_profiles_from_disk():
    """
    """
    with open('data/compas/profiles_dis2.pkl', 'rb') as f:
        profiles = pickle.load(f)

    return profiles


if __name__ == '__main__':
    """
    python -m src.cg.causal_graph_compas
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