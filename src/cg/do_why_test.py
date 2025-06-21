import pickle
from dowhy import CausalModel
import pandas as pd
import networkx as nx

import warnings
warnings.filterwarnings("ignore")  # Suppress warnings

'''
time python -m src.cg.do_why_test
'''

def get_adult_cg():
    """
    """
    # Geneate Causal Graph
    # graph taken from https://par.nsf.gov/servlets/purl/10126315
    V = ['age', 'race', 'gender', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'hours-per-week', 'native-country', 'income']
    target = 'income'
    g = nx.DiGraph()
    g.add_nodes_from(V)

    # Full graph
    g.add_edge('age', 'marital-status')
    g.add_edge('native-country', 'marital-status')
    g.add_edge('race', 'marital-status')
    g.add_edge('gender', 'marital-status')
    
    g.add_edge('age', 'education')
    g.add_edge('native-country', 'education')
    g.add_edge('race', 'education')
    g.add_edge('gender', 'education')
    g.add_edge('marital-status', 'education')

    g.add_edge('age', 'hours-per-week')
    g.add_edge('native-country', 'hours-per-week')
    g.add_edge('race', 'hours-per-week')
    g.add_edge('gender', 'hours-per-week')
    g.add_edge('education', 'hours-per-week')
    g.add_edge('marital-status', 'hours-per-week')

    g.add_edge('age', 'occupation')
    g.add_edge('marital-status', 'occupation')
    g.add_edge('race', 'occupation')
    g.add_edge('gender', 'occupation')
    g.add_edge('education', 'occupation')

    g.add_edge('age', 'workclass')
    g.add_edge('native-country', 'workclass')
    g.add_edge('marital-status', 'workclass')
    g.add_edge('education', 'workclass')

    g.add_edge('age', 'relationship')
    g.add_edge('native-country', 'relationship')
    g.add_edge('gender', 'relationship')
    g.add_edge('education', 'relationship')
    g.add_edge('marital-status', 'relationship')

    g.add_edge('age', 'income')
    g.add_edge('native-country', 'income')
    g.add_edge('race', 'income')
    g.add_edge('gender', 'income')
    g.add_edge('education', 'income')
    g.add_edge('marital-status', 'income')
    g.add_edge('hours-per-week', 'income')
    g.add_edge('workclass', 'income')
    g.add_edge('relationship', 'income')
    g.add_edge('occupation', 'income')

    return g, V, target


from src.proc_data.proc_adult import load_adult\

adult, adult_1hot, target, train_set_indices, test_indices_no_duplicate = load_adult()
train_set, test_set = adult.loc[train_set_indices], adult.loc[test_indices_no_duplicate]

g, V, target = get_adult_cg()

with open(f'data/adult/rules_freq_500_supp_0.3.pkl', 'rb') as f:
    filtered_rules_dict = pickle.load(f)

rules_list = list(filtered_rules_dict.items())

rule = rules_list[1000][1][0]

print(f"Rule: {rule}")

# print(rules_list[1000:1005])


# exit(0)

# Create Causal Model
model = CausalModel(
    data=train_set,
    treatment=['age', 'race', 'gender', 'native-country'],  # Example treatment variables
    outcome=target,
    graph=g
)

# Identify causal effect (automatically selects confounders using back-door)
identified_estimand = model.identify_effect()
print(f"Identified Estimand: {identified_estimand}")

# Estimate ATE (you can use method_name="backdoor.linear_regression", etc.)
# estimate = model.estimate_effect(identified_estimand,
#                                  method_name="backdoor.linear_regression")
estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.generalized_linear_model")


print("Estimated ATE:", estimate.value)

temp_df = train_set.query(f'not(age == 3)')
print(len(temp_df))

model = CausalModel(
    data=temp_df,
    treatment=['age', 'race', 'gender', 'native-country'],  # Example treatment variables
    outcome=target,
    graph=g
)

# Identify causal effect (automatically selects confounders using back-door)
identified_estimand = model.identify_effect()
print(f"Identified Estimand: {identified_estimand}")

# Estimate ATE (you can use method_name="backdoor.linear_regression", etc.)
# estimate = model.estimate_effect(identified_estimand,
#                                  method_name="backdoor.linear_regression")
estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.generalized_linear_model")
print("Estimated ATE:", estimate.value)
