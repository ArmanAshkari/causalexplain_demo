import warnings
warnings.filterwarnings("ignore")

import multiprocessing
from pathlib import Path
import re
import streamlit as st
import io
import pandas as pd
import pickle
import time
import pygraphviz as pgv
from networkx.drawing.nx_agraph import to_agraph
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import JsCode
from copy import deepcopy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from src.proc_data.proc_adult import load_adult
from src.proc_data.proc_so import load_so
from src.proc_data.proc_compas import load_compas, inverse_preprocess_compas
from src.proc_data.stats import read_to_shared_dict

from src.proc_data.util import get_base_predictions
from .dataframe_highlight import apply_highlight
from .util import custom_css, make_cell_style_jscode, make_cell_style_jscode2



np.random.seed(42)


def store_value(key):
    st.session_state[key] = st.session_state["_"+key]


def load_value(key):
    st.session_state["_"+key] = st.session_state[key]


def rule_to_predicate_cg(rule_w_target, one_hot=False, multi_valued=False):
    """
    """
    regex = rf"([^_]+)_(\d+)" # use this regex to match one-hot encoded features for multi-valued features
    rule_wo_target = rule_w_target[:-1]  # Exclude the target value
    target_clause = rule_w_target[-1]  # The target value

    dataset_name = st.session_state.selected_dataset
    if dataset_name == "Adult Income":
        le_dict = st.session_state.le_dict
        target = st.session_state.target      
        # rule_wo_target_converted = ' or '.join([' and '.join([`{feat}`=={val}' for feat, val in conjunction]) for conjunction in rule_wo_target])
        rule_wo_target_converted = ' or '.join([' and '.join([f"`{feat}`=='{le_dict[feat].inverse_transform([val])[0]}'" for feat, val in conjunction]) for conjunction in rule_wo_target])
        
        return f"({rule_wo_target_converted}) and `{target_clause[0]}`=='{le_dict[target].inverse_transform([target_clause[1]])[0]}'"
    
    elif dataset_name == "Stackoverflow Annual Developer Survey":
        le_dict = st.session_state.le_dict
        multi_val_col_metadata = st.session_state.multi_val_col_metadata
        mutli_val_remap = st.session_state.multi_val_remap
        target = st.session_state.target

        rule_wo_target_converted = ' or '.join([' and '.join([f"`{feat}`=='{le_dict[feat].inverse_transform([val])[0]}'" if feat in le_dict else f'`{mutli_val_remap[feat]}`=={val}' for feat, val in conjunction]) for conjunction in rule_wo_target])
        
        return f"({rule_wo_target_converted}) and `{target_clause[0]}`=='{'>=Mdn' if target_clause[1] == 1 else '<Mdn'}'"
    
    elif dataset_name == "Compas":
        target = st.session_state.target
        all_maps = st.session_state.all_maps

        rule_wo_target_converted = ' or '.join([' and '.join([f"`{feat}`=='{all_maps[feat][val]}'" for feat, val in conjunction]) for conjunction in rule_wo_target])
        
        return f"({rule_wo_target_converted}) and `{target_clause[0]}`=='{True if target_clause[1] == 1 else False}'"


def load_training_data():
    """
    """
    selected_dataset = st.session_state.selected_dataset
    
    if selected_dataset == "Adult Income":
        adult, _, target, train_set_indices, _ = load_adult()
        with open('data/adult/le_dict.pkl', 'rb') as f:
            le_dict = pickle.load(f)
    
        train_set = adult.loc[train_set_indices]
        for col in train_set.columns:
            train_set[col] = le_dict[col].inverse_transform(train_set[col])
            
        st.session_state["training_data"] = train_set
        st.session_state["target"] = target
        st.session_state["le_dict"] = le_dict
            
    elif selected_dataset == "Stackoverflow Annual Developer Survey":
        so, _, target, train_set_indices, _ = load_so()
        with open('data/so/le_dict.pkl', 'rb') as f:
            le_dict = pickle.load(f)
        with open(f'data/so/multi_val_col_metadata.pkl', 'rb') as f:
            multi_val_col_metadata = pickle.load(f)

        train_set = so.loc[train_set_indices]
        for col in le_dict.keys():
            train_set[col] = le_dict[col].inverse_transform(train_set[col])

        multi_val_remap = {}
        for col in multi_val_col_metadata:
            temp_dict = multi_val_col_metadata[col]
            for key, val in temp_dict.items():
                train_set.rename(columns={f'{col}_{val}': f'{col}_{key}'}, inplace=True)
                multi_val_remap[f'{col}_{val}'] = f'{col}_{key}'

        
        train_set[target] = train_set[target].apply(lambda x: '>=Mdn' if x == 1 else '<Mdn')
        
        st.session_state["training_data"] = train_set
        st.session_state["target"] = target
        st.session_state["le_dict"] = le_dict
        st.session_state["multi_val_col_metadata"] = multi_val_col_metadata
        st.session_state["multi_val_remap"] = multi_val_remap
        
    elif selected_dataset == "Compas":
        compas, _, target, train_set_indices, _ = load_compas()
        compas, all_maps = inverse_preprocess_compas(compas)
        compas[target] = compas[target].apply(lambda x: 'True' if x == 1 else 'False')

        train_set = compas.loc[train_set_indices]
        
        st.session_state["training_data"] = train_set
        st.session_state["target"] = target
        st.session_state["all_maps"] = all_maps


def load_test_data():
    """
    """
    selected_dataset = st.session_state.selected_dataset

    if selected_dataset == "Adult Income":
        adult, _, target, _, test_indices_no_duplicate = load_adult()
        with open('data/adult/le_dict.pkl', 'rb') as f:
            le_dict = pickle.load(f)

        test_set = adult.loc[test_indices_no_duplicate]
        st.session_state["y_true"] = test_set[target]
        for col in test_set.columns:
            test_set[col] = le_dict[col].inverse_transform(test_set[col])
        
        st.session_state["_test_data"] = test_set
        st.session_state["test_data"] = test_set.drop(columns=[target], axis=1)
        st.session_state["test_indices"] = test_indices_no_duplicate
            
    elif selected_dataset == "Stackoverflow Annual Developer Survey":
        so, _, target, _, test_indices_no_duplicate = load_so()
        with open('data/so/le_dict.pkl', 'rb') as f:
            le_dict = pickle.load(f)
        with open(f'data/so/multi_val_col_metadata.pkl', 'rb') as f:
            multi_val_col_metadata = pickle.load(f)

        test_set = so.loc[test_indices_no_duplicate]
        st.session_state["y_true"] = test_set[target]
        for col in le_dict.keys():
            test_set[col] = le_dict[col].inverse_transform(test_set[col])

        for col in multi_val_col_metadata:
            temp_dict = multi_val_col_metadata[col]
            for key, val in temp_dict.items():
                test_set.rename(columns={f'{col}_{val}': f'{col}_{key}'}, inplace=True)

        test_set[target] = test_set[target].apply(lambda x: '>=Mdn' if x == 1 else '<Mdn')
        
        st.session_state["_test_data"] = test_set
        st.session_state["test_data"] = test_set.drop(columns=[target], axis=1)
        st.session_state["test_indices"] = test_indices_no_duplicate
        
    elif selected_dataset == "Compas":
        compas, _, target, _, test_indices_no_duplicate = load_compas()
        compas, _ = inverse_preprocess_compas(compas)
        st.session_state["y_true"] = compas[target].loc[test_indices_no_duplicate]
        compas[target] = compas[target].apply(lambda x: 'True' if x == 1 else 'False')

        test_set = compas.loc[test_indices_no_duplicate]
        
        st.session_state["_test_data"] = test_set
        st.session_state["test_data"] = test_set.drop(columns=[target], axis=1)
        st.session_state["test_indices"] = test_indices_no_duplicate


def run_classification():
    """
    """
    target = st.session_state.target
    y_true = st.session_state._test_data[st.session_state.target]

    dataset_name = st.session_state.selected_dataset
    model_name = st.session_state.selected_model
    if dataset_name == "Adult Income":
        _dataset_name = 'adult'
    elif dataset_name == "Stackoverflow Annual Developer Survey":
        _dataset_name = 'so'
    elif dataset_name == "Compas":
        _dataset_name = 'compas'
    
    if model_name == "SVM":
        _model_name = 'linsvc'
    elif model_name == "Logistic Regression":
        _model_name = 'logreg'
    elif model_name == "Neural Network":
        _model_name = 'nn'
    elif model_name == "XGBoost":
        _model_name = 'xgboost'
    elif model_name == "Adaboost":
        _model_name = 'adaboost'
    elif model_name == "Random Forest":
        _model_name = 'rf'
    
    y_pred = get_base_predictions(dataset_name=_dataset_name, model_name=_model_name)
    # y_pred = st.session_state.le_dict[st.session_state.target].inverse_transform(y_pred)

    st.session_state["base_y_pred"] = y_pred

    classification_df = st.session_state.test_data.copy()
    prediction_col = f"{target} (Prediction)"
    true_value_col = f"{target} (True Value)"
    if dataset_name == 'Adult Income':
        le_dict = st.session_state.le_dict
        classification_df[prediction_col] = le_dict[target].inverse_transform(y_pred)
        classification_df[true_value_col] = y_true.values
    elif dataset_name == 'Stackoverflow Annual Developer Survey':
        classification_df[prediction_col] = y_pred
        classification_df[prediction_col] = classification_df[prediction_col].apply(lambda x: '>=Mdn' if x == 1 else '<Mdn')
        classification_df[true_value_col] = y_true.values
    elif dataset_name == 'Compas':
        classification_df[prediction_col] = y_pred
        classification_df[prediction_col] = classification_df[prediction_col].apply(lambda x: 'True' if x == 1 else 'False')
        classification_df[true_value_col] = y_true.values

    classification_df.insert(0, "index", range(len(classification_df)))
    # classification_df['index'] =  range(len(classification_df))
    # classification_df.index.name = "Index"
    # classification_result = classification_df.style.apply(apply_highlight, target=target, axis=None)

    # columns_with_border = [f'{target} (Prediction)',  f'{target} (True Value)']
    # classification_result = classification_result.set_table_styles(
    #     [
    #         {'selector': f'th:nth-child({classification_df.columns.get_loc(col) + 1})', 'props': [('border', '2px solid red')]} 
    #         for col in columns_with_border
    #     ] + [
    #         {'selector': f'td:nth-child({classification_df.columns.get_loc(col) + 1})', 'props': [('border', '2px solid red')]} 
    #         for col in columns_with_border
    #     ],
    #     overwrite=False  # Keep existing styles and add new ones
    # )

    # classification_result = classification_result.set_table_styles(
    #     [
    #         {'selector': f'th:nth-child({classification_df.columns.get_loc(f"{target} (Prediction)") + 1})', 'props': [('border', '1px solid pink')]} 
    #     ] + [
    #         {'selector': f'td:nth-child({classification_df.columns.get_loc(f"{target} (Prediction)") + 1})', 'props': [('border', '1px solid pink')]} 
    #     ] + [
    #         {'selector': f'th:nth-child({classification_df.columns.get_loc(f"{target} (True Value)") + 1})', 'props': [('border', '1px solid lightgreen')]} 
    #     ] + [
    #         {'selector': f'td:nth-child({classification_df.columns.get_loc(f"{target} (True Value)") + 1})', 'props': [('border', '1px solid lightgreen')]} 
    #     ],
    #     overwrite=False  # Keep existing styles and add new ones
    # )

    # styled_df = styled_df.set_table_styles(
    # [
    #     {'selector': f'th:nth-child({df.columns.get_loc(col) + 1})', 'props': [('border', '2px solid red')]} 
    #     for col in columns_with_border
    # ] + [
    #     {'selector': f'td:nth-child({df.columns.get_loc(col) + 1})', 'props': [('border', '2px solid red')]} 
    #     for col in columns_with_border
    # ],
    # overwrite=False  # Keep existing styles and add new ones
    # )

    # classification_result.style.apply(apply_highlight, axis=None)

    # st.session_state["classification_result"] = classification_result
    st.session_state["classification_result"] = classification_df

    # st.write(type(classification_result))


def load_causal_graph():
    """
    """
    # if 'CG' not in st.session_state:
    #     st.write("⚠️ **Causal DAG not found! Please load the graph first.**")
    #     return
    
    dataset_name = st.session_state.selected_dataset
    if dataset_name == "Adult Income":
        CG = nx.read_edgelist("demo_ui/data/adult_cg.txt", create_using=nx.DiGraph(), nodetype=str)
        st.session_state.CG = CG
    elif dataset_name == "Stackoverflow Annual Developer Survey":
        CG = nx.read_edgelist("demo_ui/data/so_cg.txt", create_using=nx.DiGraph(), nodetype=str)
        st.session_state.CG = CG
    elif dataset_name == "Compas":
        CG = nx.read_edgelist("demo_ui/data/compas_cg.txt", create_using=nx.DiGraph(), nodetype=str)
        st.session_state.CG = CG

    A = to_agraph(st.session_state.CG)
    target_column = st.session_state.target
    # Special highlight for target node 'income': white with red border
    A.get_node(target_column).attr.update(color='red', fontcolor='red', fillcolor='white', style='filled', shape='ellipse')
    buf = io.BytesIO()
    A.draw(buf, format='png', prog='dot')
    st.session_state["_CG_buf"] = buf
    st.session_state["CG_buf"] = buf
    st.session_state["CG_buf2"] = buf
    st.success("✅ Causal DAG Loaded!")


def reload_causal_graph():
    """
    """
    if 'CG' not in st.session_state:
        # st.write("⚠️ **Causal Graph not found! Please load the graph first.**")
        return
    
    st.session_state.CG_buf = st.session_state._CG_buf
    st.session_state.CG_buf2 = st.session_state._CG_buf


def highlighted_graph(profile):
    """
    """
    g = deepcopy(st.session_state.CG)
    highlight_edges = set()
    highlight_nodes = set()

    for conjunction in profile:
        path_to_highlight = list()
        for i in range(len(conjunction)-1):
            # Find paths
            path_to_highlight.extend(list(nx.all_simple_paths(g, source=conjunction[i], target=conjunction[i+1])))
            highlight_nodes.add(conjunction[i])
        
        path_to_highlight.extend(list(nx.all_simple_paths(g, source=conjunction[-1], target=st.session_state.target)))
        highlight_nodes.add(conjunction[-1])

        for path in path_to_highlight:
            highlight_edges.update(zip(path[:-1], path[1:]))

    # Create a copy of the graph for visualization
    A = nx.drawing.nx_agraph.to_agraph(g)

    # Set default node attributes: white nodes with black border
    A.node_attr.update(color='black', fillcolor='white', style='filled', shape='ellipse')

    # Highlight specific nodes: light blue with black border
    for node in highlight_nodes - {'income'}:
        A.get_node(node).attr.update(color='black', fillcolor='lightblue', style='filled', shape='ellipse')

    # Special highlight for target node 'income': white with red border
    A.get_node(st.session_state.target).attr.update(color='red', fillcolor='white', style='filled', shape='ellipse')

    # Highlight specific edges: red edges
    for edge in highlight_edges:
        A.get_edge(edge[0], edge[1]).attr.update(color='red', penwidth=3)
    
    return A


def profile_from_rule(rule):
    """
    """
    rule_wo_target = rule[:-1]  # Exclude the target value
    dataset_name = st.session_state.selected_dataset
    if dataset_name == "Stackoverflow Annual Developer Survey":        
        regex = rf"([^_]+)_(\d+)"
        rule_wo_target_converted = tuple([tuple([re.match(regex, feat).group(1) if re.match(regex, feat) 
                                                  else feat for feat, _ in conjunction]) for conjunction in rule_wo_target ])
    else:
        rule_wo_target_converted = tuple([tuple([feat for feat, _ in conjunction]) for conjunction in rule_wo_target ])
    return rule_wo_target_converted


def generate_profiles():
    """Generates a list of profile items and stores them as a Pandas DataFrame in session state."""
    # st.session_state["profiles"] = pd.DataFrame({"Profiles": [f"Profile {i}" for i in range(1, 21)]})  # 20 Profile Names

    dataset_name = st.session_state.selected_dataset
    if dataset_name == "Adult Income":
        with open(f'data/adult/profiles_dis2.pkl', 'rb') as f:
            _profiles = pickle.load(f)
    elif dataset_name == "Stackoverflow Annual Developer Survey":
        with open(f'data/so/profiles_dis2.pkl', 'rb') as f:
            _profiles = pickle.load(f)
    elif dataset_name == "Compas":
        with open(f'data/compas/profiles_dis2.pkl', 'rb') as f:
            _profiles = pickle.load(f)
    
    profiles = [' or '.join(['(' + ' and '.join([f'`{c}`==*' for c in conjunction]) + ')' for conjunction in profile]) for profile in _profiles]

    st.session_state["_profiles"] = _profiles
    st.session_state["profiles"] = pd.DataFrame({"Profiles": profiles})
            

def filter_rules():
    """
    """
    dataset_name = st.session_state.selected_dataset
    if dataset_name == "Adult Income":
        rules_dir = f'data/adult/rules_freq_2000_supp_0.3_w_ATE.pkl'
    elif dataset_name == "Stackoverflow Annual Developer Survey":
        rules_dir = f'data/so/rules_freq_1000_supp_0.3_w_ATE.pkl'
    elif dataset_name == "Compas":
        rules_dir = f'data/compas/rules_freq_1000_supp_0.3_w_ATE.pkl'

    with open(rules_dir, 'rb') as f:
        rules_dict = pickle.load(f)

    st.session_state["rules_dict"] = rules_dict
    
    _rules = [(rule, support, del_ATE) for rule, support, del_ATE in rules_dict.values()]
    _rules_temp = [(rule_to_predicate_cg(rule), support, del_ATE) for rule, support, del_ATE in rules_dict.values()]
    rules = pd.DataFrame(data=_rules_temp, columns=['Rules', 'Support', 'ATE'])
    # rules['Δ ATE'] = np.random.uniform(5, 10, len(rules))

    st.session_state["_rules"] = _rules
    st.session_state["rules"] = rules    


def process_metadata():
    """
    """
    dataset_name = st.session_state.selected_dataset
    model_name = st.session_state.selected_model

    if dataset_name == "Adult Income":
        _dataset_name = 'adult'
    elif dataset_name == "Stackoverflow Annual Developer Survey":
        _dataset_name = 'so'
    elif dataset_name == "Compas":
        _dataset_name = 'compas'
    
    if model_name == "SVM":
        _model_name = 'linsvc'
    elif model_name == "Logistic Regression":
        _model_name = 'logreg'
    elif model_name == "Neural Network":
        _model_name = 'nn'
    elif model_name == "XGBoost":
        _model_name = 'xgboost'
    elif model_name == "Adaboost":
        _model_name = 'adaboost'
    elif model_name == "Random Forest":
        _model_name = 'rf'

    with open(f'data/{_dataset_name}/{_model_name}_base_pred.pkl', 'rb') as f:
        base_y_pred = pickle.load(f)

    flip_history = {i:set() for i in range(len(base_y_pred))}
    # highest_model_sim = {i:None for i in range(len(base_y_pred))}

    # with open(rules_dir, 'rb') as f:
    #     rules_dict = pickle.load(f)

    rules_dict = st.session_state.rules_dict
    
    rules_list = list(rules_dict.items())

    # filenames = [f'{global_index}_{label}.pkl' for (global_index, label), (rule_w_target, support, del_ATE) in rules_list]
    filenames = [f'{global_index}_{label}.pkl' for global_index, label in rules_dict.keys()]
    directory = f'data/{_dataset_name}/saved_models/{_model_name}/metadata'

    temp = list()
    for filename in filenames:
        file_path = Path(f'{directory}/{filename}')
        with open(file_path, 'rb') as f:
            data = pickle.load(f)  # Read the .pkl file
        temp.append(data)

    metadata = {}
    for i, (global_index, label) in enumerate(rules_dict.keys()):
        metadata[(global_index, label)] = temp[i]

    for (global_index, label), (accuracy, model_similarity, new_y_pred) in metadata.items():
        flips = (base_y_pred != new_y_pred).astype(int)
        flips_where = np.where(flips)[0]
        for j in flips_where:
            flip_history[j].add((global_index, label))
    
    # with multiprocessing.Manager() as manager:
    #     metadata = manager.dict()

    #     read_to_shared_dict(metadata, directory, filenames, max_workers=8, batch_size=1000)
    #     print(f'{len(metadata)} files read into shared dictionary')

    #     for (global_index, label), (accuracy, model_similarity, new_y_pred) in metadata.items():

    #         flips = (base_y_pred != new_y_pred).astype(int)
    #         flips_where = np.where(flips)[0]
    #         for j in flips_where:
    #             flip_history[j].add((global_index, label))
    #             # if highest_model_sim[j] is None or model_similarity > highest_model_sim[j][1]:
    #             #     highest_model_sim[j] = ((global_index, label), model_similarity)

    st.session_state["flip_history"] = flip_history
    st.session_state["metadata"] = dict(metadata)


def get_exp_set(index):
    """
    """
    flip_history = st.session_state.flip_history
    rules_dict = st.session_state.rules_dict
    metadata = st.session_state.metadata

    all_exp_set = []

    for global_index, label in flip_history[index]:
        rule_w_target, support, del_ATE = rules_dict[(global_index, label)]
        accuracy, model_similarity, new_y_pred = metadata[(global_index, label)]

        all_exp_set.append((rule_w_target, support, del_ATE, accuracy, model_similarity))
    
    all_exp_set.sort(key=lambda x: (-x[4], x[1], -x[3]))

    return all_exp_set


def go_next():
    """Increment step in session state."""
    if st.session_state.step < 7:
        st.session_state.step += 1
    
    if st.session_state.step in [3,4]:
        reload_causal_graph()

def go_previous():
    """Decrement step in session state."""
    if st.session_state.step > 1:
        st.session_state.step -= 1
    
    if st.session_state.step in [3,4]:
        reload_causal_graph()


def render_main_pane(steps):
    """
    """
    # Main Content
    st.title("CausalExplain: Causal Explanation for Black-box Models")
    st.subheader(f"Step {st.session_state.step}: {steps[st.session_state.step]}")

    # Display session state values
    # for _ in ['step', 'selected_model', 'conj_max_len', 'disj_max_len', 'num_exp_set', 'support_thrs', 'delta_ATE', 'num_exp_set']:
    #     st.write(_, st.session_state[_])

    # for _ in ['profiles', 'rules', '_profiles', '_rules']:
    #     if _ in st.session_state:
    #         st.write(_, len(st.session_state[_]))


    if st.session_state.step == 1:
        st.write("Welcome to the CausalExplain demo! This application allows you to explore causal explanations for black-box models using various datasets and machine learning models.")
        st.write("📂 First, select a Dataset from the sidebar and Load the Training and Test Data. This will show you a preview of the dataset.")
        st.write("The target variable is the last column in the dataset and highlighted in red, which is used for classification.")
        st.write("After loading the data, you can proceed to the next step to choose a machine learning model and train it for classification.")
        

        if st.button("📥 Load Training Data"):
            # load_dummy_csv("training_data")
            load_training_data()
            st.success("✅ Training data loaded!")

        if st.button("📥 Load Test Data"):
            # load_dummy_csv("test_data")
            load_test_data()
            st.success("✅ Test data loaded!")

        # if st.session_state.training_data:
        if 'training_data' in st.session_state:
            st.write("📊 **Training Data Preview:**")
            # st.dataframe(pd.read_csv(io.StringIO(st.session_state.training_data)))
            # st.dataframe(st.session_state.training_data, hide_index=True, height=250)
            data = st.session_state.training_data
            gb = GridOptionsBuilder.from_dataframe(data)
            gb.configure_selection('single')

            for col in data.columns:
                gb.configure_column(
                    col, 
                    headerTooltip=col,  # Tooltip to show full column name
                    width=300,          # Increase width for visibility
                    wrapHeaderText=True,  # Wrap text in header
                    autoHeaderHeight=True # Auto-adjust header height
                )

            from .util import custom_css, make_cell_style_jscode, make_cell_style_jscode2

            target = st.session_state.target

            gb.configure_column(target, headerClass="prediction-header", cellStyle=make_cell_style_jscode(target), pinned='right')
            # gb.configure_column("income", headerClass="truevalue-header", cellStyle=cell_style_jscode2)

            grid_options = gb.build()

            # Create narrow column to shrink the grid
            col1, col2 = st.columns([0.9, 0.1])  # Adjust column width ratio

            with col1:
                # st.subheader("Select Test Row:")
                grid_response = AgGrid(
                    data,
                    gridOptions=grid_options,
                    fit_columns_on_grid_load=False,
                    height=250,
                    enable_enterprise_modules=False,
                    allow_unsafe_jscode=True,
                    custom_css=custom_css
                )

        # if st.session_state.test_data:
        if 'test_data' in st.session_state:
            st.write("📊 **Test Data Preview:**")
            # st.dataframe(pd.read_csv(io.StringIO(st.session_state.test_data)))
            # st.dataframe(st.session_state.test_data, hide_index=True, height=250)
            data = st.session_state.test_data
            gb = GridOptionsBuilder.from_dataframe(data)
            gb.configure_selection('single')

            for col in data.columns:
                gb.configure_column(
                    col, 
                    headerTooltip=col,  # Tooltip to show full column name
                    width=300,          # Increase width for visibility
                    wrapHeaderText=True,  # Wrap text in header
                    autoHeaderHeight=True # Auto-adjust header height
                )

            from .util import custom_css, make_cell_style_jscode, make_cell_style_jscode2

            # gb.configure_column("income (Prediction)", headerClass="prediction-header", cellStyle=cell_style_jscode)
            # gb.configure_column("income (True Value)", headerClass="truevalue-header", cellStyle=cell_style_jscode2)

            grid_options = gb.build()

            # Create narrow column to shrink the grid
            col1, col2 = st.columns([0.9, 0.1])  # Adjust column width ratio

            with col1:
                # st.subheader("Select Test Row:")
                grid_response = AgGrid(
                    data,
                    gridOptions=grid_options,
                    fit_columns_on_grid_load=False,
                    height=250,
                    enable_enterprise_modules=False,
                    allow_unsafe_jscode=True,
                    custom_css=custom_css
                )

    elif st.session_state.step == 2:
        st.write("⚙️ Next, choose a machine learning model from the sidebar, train the model and run classification. This will show you the classification results.")
        st.write("The classification results will be displayed in a table, where predictions and true values for the target variable are highlighted in red and green. The misclassified prediction will be highlighted in solid.")
        st.write("You can then proceed to the next step to load a causal graph and generate Feature Profiles.")

        if st.button("🚀 Train Model"):
            with st.spinner(f'Training {st.session_state.selected_model} Model...'):
                if st.session_state.selected_model == "Neural Network":
                    time.sleep(10)
                else:
                    time.sleep(3)
            st.success("✅ Training finished!")  # Placeholder for future logic

        if st.button("📊 Run Classification"):
            # load_dummy_csv("classification_results")
            run_classification()
            # st.dataframe(st.session_state.classification_result, height=400)

            if 'classification_result' in st.session_state:
                data = st.session_state.classification_result
                gb = GridOptionsBuilder.from_dataframe(data)
                gb.configure_selection('single')

                for col in data.columns:
                    gb.configure_column(
                        col, 
                        headerTooltip=col,  # Tooltip to show full column name
                        width=300,          # Increase width for visibility
                        wrapHeaderText=True,  # Wrap text in header
                        autoHeaderHeight=True # Auto-adjust header height
                    )

                from .util import custom_css, make_cell_style_jscode, make_cell_style_jscode2

                target = st.session_state.target

                gb.configure_column(f"{target} (Prediction)", headerClass="prediction-header", cellStyle=make_cell_style_jscode(target), pinned='right')
                gb.configure_column(f"{target} (True Value)", headerClass="truevalue-header", cellStyle=make_cell_style_jscode2(target), pinned='right')

                grid_options = gb.build()

                # Create narrow column to shrink the grid
                col1, col2 = st.columns([0.9, 0.1])  # Adjust column width ratio

                with col1:
                    st.subheader("Select Test Row:")
                    grid_response = AgGrid(
                        data,
                        gridOptions=grid_options,
                        fit_columns_on_grid_load=False,
                        height=250,
                        enable_enterprise_modules=False,
                        allow_unsafe_jscode=True,
                        custom_css=custom_css
                    )
                st.success("✅ Classification results generated!")


    elif st.session_state.step == 3:
        st.write("⚙️ **Load Causal DAG ➡ Set Parameters ➡ Generate Profiles**")
        st.write("Load the causal graph first. The causal graph is a directed acyclic graph (DAG) that represents the causal relationships between features in the dataset. It is used to generate profiles, which are sets of of features causally related to the target variable expressed in disjunctive normal form, for example, `((A1 ∧ A2) ∨ (B1 ∧ B2))`.")
        st.write("Set the parameters for generating profiles, such as the maximum length of conjunctions and disjunctions. These parameters will be used to generate profiles based on the causal graph.") 
        st.write("Press the button to generate profiles based on the causal graph and the parameters set. The generated profiles will be displayed in a table below.")
        st.write("After generating profiles, you can select a profile to show the causal paths it supports. The causal paths are highlighted in the causal graph.")
        st.write("You can now proceed to the next step to generate filtered rules.")

        col1, col2 = st.columns(2)

        # Left Column: Load and Display Causal Graph
        with col1:
            if st.button("📌 Load Causal DAG"):
                # load_dummy_image()
                load_causal_graph()
                

            if 'CG_buf' in st.session_state:
                # st.image(st.session_state["causal_graph"], caption="Causal DAG", use_container_width=True)
                st.write("📄 **Causal DAG Preview**")
                st.image(st.session_state.CG_buf, use_container_width=True)

        # Right Column: Generate Profiles and Display Selectable Table
        with col2:
            if st.button("🛠️ Generate Profiles"):
                generate_profiles()
                st.success("✅ Profiles Generated!")

            if not st.session_state.profiles.empty:
                st.write("📜 **Select a Profile to show the Causal Paths it supports**")
                st.write(f"{len(st.session_state.profiles)} profiles generated. The *s in the profiles are placeholders.")

                profiles = st.session_state.profiles
                profiles.index.name = "Index"
                profiles = profiles.reset_index()

                gb = GridOptionsBuilder.from_dataframe(profiles)
                gb.configure_selection(selection_mode="single")  # Select entire row on click
                # gb.configure_pagination(paginationAutoPageSize=True)  # Enable pagination
                # gb.configure_column("Index", pinned="left")  # Pin index column to the left
                gb.configure_column('Profiles', width=1000, wrapText=True, autoHeight=True)  # Automatically adjust width
                grid_options = gb.build()

                grid_response = AgGrid(profiles,
                                    gridOptions=grid_options,
                                    height=350,
                                    fit_columns_on_grid_load=True,
                                    enable_enterprise_modules=False)

                # Get selected row
                selected_rows = grid_response["selected_rows"]

                # Display selected row
                if selected_rows is not None and len(selected_rows) > 0:
                    st.write("Selected Profile:")
                    st.write(selected_rows['Index'].values[0], selected_rows['Profiles'].values[0])

                    if st.session_state.step3_selected_row != selected_rows['Profiles'].values[0]:
                        h_CG = highlighted_graph(st.session_state._profiles[selected_rows['Index'].values[0]])
                        # print(st.session_state._profiles[selected_rows['Index'].values[0]])

                        import io
                        buf = io.BytesIO()
                        h_CG.draw(buf, format='png', prog='dot')
                        st.session_state["CG_buf"] = buf
                        st.session_state.step3_selected_row = selected_rows['Profiles'].values[0]
                        st.rerun()

    elif st.session_state.step == 4:
        st.write("⚙️ **Set Parameters ➡ Filter Rules**")
        st.write("Rules are instantiated from the profiles and feature values in the training data. They describe subsets in the training data.")
        st.write("Explanation sets are also training data subsets that, when removed, can change the prediction for a test row. Therefore, rules are used to desribe explanation sets.")
        st.write("However, other test rows are also affected by explanation sets, causing side effects. Smaller explanation sets minimize the side effects.")
        st.write("Support is the fraction of the explanation set size to the full training data. Max support allows to filter out explanation sets that are too large.") 
        st.write("The system also uses the Average Treatment Effect (ATE) to identify causally important rules. The Min ATE is a threshold to identify important rules. Rules with ATE below the threshold are filtered out.")
        st.write("Press the button to generate filtered rules based on the profiles and the parameters set. The generated rules will be displayed in a table below. You can select a profile to show the causal paths it supports. The causal paths are highlighted in the causal graph.")
        st.write("After generating filtered rules, you can proceed to the next step to review the rules and retrain the model.")

        col1, col2 = st.columns(2)

        # Left Column: Display Image (No Buttons)
        with col1:
            if 'CG_buf2' in st.session_state:
                st.write("📄 **Causal DAG Preview**")
                st.image(st.session_state.CG_buf2, use_container_width=True)

        # Right Column: Display Selectable Table
        with col2:
            if st.button("📋 Generate Filtered Rules"):
                # generate_double_column_table()
                filter_rules()
                st.success("✅ Filtered Rules Generated!")

            if not st.session_state.rules.empty:
                st.write("📜 **Select a Rules to show the Causal Paths it supports**")
                st.write(f"{len(st.session_state.rules)} rules generated.")

                rules = st.session_state.rules
                rules.index.name = "Index"
                rules = rules.reset_index()

                hist_support = [support for _, support,_ in st.session_state._rules]
                fig, ax = plt.subplots(figsize=(8, 3), dpi=400)
                ax.hist(hist_support, bins=20, color='skyblue', edgecolor='black')
                ax.set_title('Histogram of Rule Support')
                ax.set_xlabel('Support')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)

                gb = GridOptionsBuilder.from_dataframe(rules)
                gb.configure_selection(selection_mode="single")  # Select entire row on click
                # gb.configure_pagination(paginationAutoPageSize=True)  # Enable pagination
                # gb.configure_column("Index", pinned="left")  # Pin index column to the left
                # gb.configure_column('Rules', width=100)
                gb.configure_column('Rules', width=1100, auto_size=True, wrapText=True, autoHeight=True)  # Automatically adjust width
                gb.configure_column('Support', flex=1, valueFormatter="(value != null) ? value.toFixed(3) : ''" )  # Automatically adjust width
                gb.configure_column('ATE', flex=1, valueFormatter="(value != null) ? value.toFixed(3) : ''" )  # Automatically adjust width
                grid_options = gb.build()

                grid_response = AgGrid(rules,
                                    gridOptions=grid_options,
                                    height=350,
                                    fit_columns_on_grid_load=True,
                                    enable_enterprise_modules=False)

                # Get selected row
                selected_rows = grid_response["selected_rows"]

                # Display selected row
                if selected_rows is not None and len(selected_rows) > 0:
                    # print(selected_rows)
                    st.write("Selected Profile:")
                    st.write(selected_rows['Index'].values[0], selected_rows['Rules'].values[0])
                    

                    if st.session_state.step4_selected_row != selected_rows['Rules'].values[0]:
                        h_CG = highlighted_graph(profile_from_rule(st.session_state._rules[selected_rows['Index'].values[0]][0]))
                        # print(st.session_state._rules[selected_rows['Index'].values[0]])
                        import io
                        buf = io.BytesIO()
                        h_CG.draw(buf, format='png', prog='dot')
                        st.session_state["CG_buf2"] = buf
                        st.session_state.step4_selected_row = selected_rows['Rules'].values[0]
                        st.rerun()

    elif st.session_state.step == 5:
        st.subheader("Final Review & Retrain")
        st.write("This is the final step of the offline preprocessing. It will launch the retraining of the model with the selected dataset and model and set of filtered rules.")
        st.write("The retraining will take some time, so please be patient. After the retraining is finished, the system will display the histograms of model similarity and accuracy, as well as the scatter plot of fliprate vs support.")
        st.write("You can then proceed to the next step to generate explanations.")

        # Single Button in Step 5
        if st.button("✅ Confirm and Proceed"):
            with st.spinner(f'Retraining...'):
                time.sleep(5)
            process_metadata()  # Process metadata for the selected dataset and model
            st.success("✅ Retraining finished!")  # Placeholder for future logic

            metadata = st.session_state.metadata
            rule_dict = st.session_state.rules_dict

            hist_model_similarity = [model_similarity for _, model_similarity, _ in metadata.values()]
            hist_accuracy = [accuracy for accuracy, _, _ in metadata.values()]

            y_true = st.session_state.y_true
            base_y_pred = st.session_state.base_y_pred
            base_accuracy = (y_true == base_y_pred).mean()

            x, y = list(), list()
            for golbal_index, label in rule_dict.keys():
                _, support, _ = rule_dict[(golbal_index, label)]
                _, model_similarity, _ = metadata[(golbal_index, label)]
                x.append(support)
                y.append(1 - model_similarity)

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(8, 3), dpi=400)
                ax.hist(hist_model_similarity, bins=20, color='skyblue', edgecolor='black')
                ax.set_title('Histogram of Model Similarity')
                ax.set_xlabel('Model Similarity')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots(figsize=(8, 3), dpi=400)
                ax.hist(hist_accuracy, bins=20, color='skyblue', edgecolor='black')
                ax.axvline(base_accuracy, color='red', linestyle='dashed', linewidth=1, label='Base Accuracy')
                ax.set_title('Histogram of Accuracy')
                ax.set_xlabel('Accuracy')
                ax.set_ylabel('Frequency')
                ax.legend()
                st.pyplot(fig)
            with col3:
                fig, ax = plt.subplots(figsize=(8, 3), dpi=400)
                ax.scatter(x, y, color='skyblue', edgecolor='black')
                ax.set_title('Fliprate vs Support')
                ax.set_xlabel('Support')
                ax.set_ylabel('Fliprate')
                st.pyplot(fig)


    # Main Content for Step 6
    elif st.session_state.step == 6:
        # st.subheader("⚙️ Running the analysis... Please wait.")
        
        metadata = st.session_state.metadata
        rule_dict = st.session_state.rules_dict

        hist_model_similarity = [model_similarity for _, model_similarity, _ in metadata.values()]
        hist_accuracy = [accuracy for accuracy, _, _ in metadata.values()]

        y_true = st.session_state.y_true
        base_y_pred = st.session_state.base_y_pred
        base_accuracy = (y_true == base_y_pred).mean()

        x, y = list(), list()
        for golbal_index, label in rule_dict.keys():
            _, support, _ = rule_dict[(golbal_index, label)]
            _, model_similarity, _ = metadata[(golbal_index, label)]
            x.append(support)
            y.append(1 - model_similarity)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(8, 3), dpi=400)
            ax.hist(hist_model_similarity, bins=20, color='skyblue', edgecolor='black')
            ax.set_title('Histogram of Model Similarity')
            ax.set_xlabel('Model Similarity')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(8, 3), dpi=400)
            ax.hist(hist_accuracy, bins=20, color='skyblue', edgecolor='black')
            ax.axvline(base_accuracy, color='red', linestyle='dashed', linewidth=1, label='Base Accuracy')
            ax.set_title('Histogram of Accuracy')
            ax.set_xlabel('Accuracy')
            ax.set_ylabel('Frequency')
            ax.legend()
            st.pyplot(fig)
        with col3:
            fig, ax = plt.subplots(figsize=(8, 3), dpi=400)
            ax.scatter(x, y, color='skyblue', edgecolor='black')
            ax.set_title('Fliprate vs Support')
            ax.set_xlabel('Support')
            ax.set_ylabel('Fliprate')
            st.pyplot(fig)

        st.write("⚙️ **Select a Test row ➡ Select Number of Explanation Sets ➡ Press Generate Explanation Sets ➡ Select Explanation**")
        st.write("Select a test row from the classification results table and the number of Explanation Sets to generate from the sidebar and generate Explanation Sets.")
        st.write("The Explanation Sets are displayed as Expalantion Rules. Select an Explanation Rule to show the Explanation Set and the Causal Paths it supports. The Causal Paths are highlighted in the causal graph.")
        st.write("The Support, Model Similarity, Accuracy, and Average Treatment Effect (ATE) are also displayed for the Explanation Rule.")

        if 'classification_result' in st.session_state:
            data = st.session_state.classification_result
            gb = GridOptionsBuilder.from_dataframe(data)
            gb.configure_selection('single')

            # Ensure full column names are shown with tooltip and width
            for col in data.columns:
                gb.configure_column(
                    col, 
                    headerTooltip=col,  # Tooltip to show full column name
                    width=300,          # Increase width for visibility
                    wrapHeaderText=True,  # Wrap text in header
                    autoHeaderHeight=True # Auto-adjust header height
                )

            from .util import custom_css, make_cell_style_jscode, make_cell_style_jscode2

            target = st.session_state.target
            # Configure columns with custom styles

            gb.configure_column(f"{target} (Prediction)", headerClass="prediction-header", cellStyle=make_cell_style_jscode(target), pinned='right')
            gb.configure_column(f"{target} (True Value)", headerClass="truevalue-header", cellStyle=make_cell_style_jscode2(target), pinned='right')

            grid_options = gb.build()

            # Create narrow column to shrink the grid

            st.subheader("Select Test Row:")
            grid_response = AgGrid(
                data,
                gridOptions=grid_options,
                fit_columns_on_grid_load=False,
                height=250,
                enable_enterprise_modules=False,
                allow_unsafe_jscode=True,
                custom_css=custom_css
            )

            _selected_rows = grid_response['selected_rows']

            if _selected_rows is not None and len(_selected_rows) > 0:
                if st.session_state.step6_selected_row != _selected_rows['index'].values[0]:
                    st.session_state.step6_exp_set_button = 1
                    st.session_state.step6_selected_row = _selected_rows['index'].values[0]

            if st.session_state.step6_exp_set_button != 1:
                selected_rows = _selected_rows
            else:
                selected_rows = None

            # selected_rows = grid_response['selected_rows']

            if selected_rows is not None and len(selected_rows) > 0:
                
                test_indices = st.session_state.test_indices
                # profiles = st.session_state._profiles
                rules = st.session_state._rules

                # selected_index = test_indices[int(selected_rows.index[0])]
                selected_index = int(selected_rows['index'].values[0])

                all_exp_set = get_exp_set(selected_index)
                num_exp_set = st.session_state.num_exp_set
                
                if num_exp_set > len(all_exp_set):
                    _num_exp_set = len(all_exp_set)              
                else:
                    _num_exp_set = num_exp_set
                selected_exp = all_exp_set[:_num_exp_set]


                row = selected_rows.iloc[[0]]

                # _profiles = [profiles[choice] for choice in choices]
                # selected_rules = [rules[choice][0] for choice in choices]
                selected_rules = [exp[0] for exp in selected_exp]
                
                st.subheader("Explanation Rules:")

                # explanations = [' or '.join(['(' + ' and '.join([f'`{c}`=="{row.iloc[0][c]}"' for c in conjunction + ')']) for conjunction in profile])for profile in _profiles]
                # explanations = [' or '.join(['(' + ' and '.join([f'`{c}`=="{row.iloc[0][c]}"' for c in conjunction]) + ')' for conjunction in profile]) for profile in _profiles]
                explanations = [rule_to_predicate_cg(rule) for rule in selected_rules] 
                
                for explanation in explanations:
                    if st.button(explanation):
                        selected_i = None
                        for i in range(len(explanations)):

                            if explanation == explanations[i]:
                                # profile = profiles[choice]
                                selected_i = i
                                break

                        selected_rule = selected_rules[selected_i]
                        delta_ATE = selected_exp[selected_i][2]
                        accuracy = selected_exp[selected_i][3]
                        model_similarity = selected_exp[selected_i][4]
                            

                        # row = st.session_state['selected_row']
                        # predicate = ' or '.join([' and '.join([f'`{c}`=="{row.iloc[0][c]}"' for c in conjunction]) for conjunction in profile])
                        predicate = rule_to_predicate_cg(selected_rule)


                        # st.write(f"{explanation}: {predicate}")

                        train_df = st.session_state.training_data
                        exp_set = train_df.query(predicate)
                        nrows = len(exp_set)
                        suport = nrows/len(st.session_state.training_data)
                        
                        col1, col2 = st.columns([0.6, 0.4])

                        with col1:
                            st.subheader(f"Explanation Set: {nrows} rows, Support {suport:.3f}, ATE: {delta_ATE:.3f}, Accuracy: {accuracy:.3f}, Model Similarity: {model_similarity:.3f}")

                            st.dataframe(exp_set, height=200)

                        with col2:
                            st.subheader("Causal Paths Supported by Explanation")
                            import pygraphviz as pgv
                            import io
                            A = highlighted_graph(profile=profile_from_rule(selected_rule))
                            buf = io.BytesIO()
                            A.draw(buf, format='png', prog='dot')
                            st.image(buf)

    else:
        pages = {
            5: "✅ Review all selections before proceeding.",
            6: "🚀 Running the analysis...",
            7: "📊 View the results of the analysis."
        }
        st.write(pages[st.session_state.step])

    # Navigation Buttons (Previous / Next)
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.session_state.step > 1:
            st.button("⬅ Previous", on_click=go_previous)

    with col3:
        if st.session_state.step < len(steps):
            st.button("Next ➡", on_click=go_next)