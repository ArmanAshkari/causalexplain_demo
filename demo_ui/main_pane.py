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

from src.proc_data.proc_adult import load_adult
from src.proc_data.proc_so import load_so
from src.proc_data.proc_compas import load_compas

from src.proc_data.util import get_base_predictions, rule_to_predicate_cg
from .dataframe_highlight import apply_highlight
from .util import custom_css, cell_style_jscode, cell_style_jscode2

np.random.seed(42)


# custom_css = {
#     ".prediction-header": {
#         # "background-color": "red !important",
#         # "color": "white !important",
#         "font-weight": "bold !important",
#         "text-align": "center",
#         'border': '1px solid pink'
#     },
#     ".truevalue-header": {
#         # "background-color": "red !important",
#         # "color": "white !important",
#         "font-weight": "bold !important",
#         "text-align": "center",
#         'border': '1px solid lightgreen'
#     }

# }

# cell_style_jscode = JsCode("""
#     function(params) {
#         if (params.data && params.data["income (Prediction)"] !== params.data["income (True Value)"]) {
#             return { backgroundColor: "pink", 'border': '1px solid pink'};
#         }
#         return {'border': '1px solid pink'};
#     }   
# """)

# cell_style_jscode2 = JsCode("""
#     function(params) {
#         if (params.data && params.data["income (Prediction)"] !== params.data["income (True Value)"]) {
#             return { backgroundColor: "lightgreen", 'border': '2px solid lightgreen'};
#         }
#         return {'border': '1px solid lightgreen'};
#     }   
# """)


def store_value(key):
    st.session_state[key] = st.session_state["_"+key]


def load_value(key):
    st.session_state["_"+key] = st.session_state[key]


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
        pass
        
    elif selected_dataset == "Compas":
        pass

def load_test_data():
    """
    """
    selected_dataset = st.session_state.selected_dataset

    if selected_dataset == "Adult Income":
        adult, _, target, _, test_indices_no_duplicate = load_adult()
        with open('data/adult/le_dict.pkl', 'rb') as f:
            le_dict = pickle.load(f)

        test_set = adult.loc[test_indices_no_duplicate]
        for col in test_set.columns:
            test_set[col] = le_dict[col].inverse_transform(test_set[col])
        
        st.session_state["_test_data"] = test_set
        st.session_state["test_data"] = test_set.drop(columns=[target], axis=1)
        st.session_state["test_indices"] = test_indices_no_duplicate
            
    elif selected_dataset == "Stackoverflow Annual Developer Survey":
        pass
        
    elif selected_dataset == "Compas":
        pass


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
    y_pred = st.session_state.le_dict[st.session_state.target].inverse_transform(y_pred)

    classification_df = st.session_state.test_data.copy()
    prediction_col = f"{target} (Prediction)"
    true_value_col = f"{target} (True Value)"
    classification_df[prediction_col] = y_pred
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
    rule_wo_target_converted = tuple([ tuple([feat for feat, _ in conjunction]) for conjunction in rule_wo_target ])
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
        pass
    
    profiles = [' or '.join(['(' + ' and '.join([f'`{c}`==*' for c in conjunction]) + ')' for conjunction in profile]) for profile in _profiles]

    st.session_state["_profiles"] = _profiles
    st.session_state["profiles"] = pd.DataFrame({"Profiles": profiles})
            

def filter_rules():
    """
    """
    dataset_name = st.session_state.selected_dataset
    if dataset_name == "Adult Income":
        # rules_dir = f'data/adult/rules_freq_2000_supp_0.3.pkl'
        rules_dir = f'data/adult/rules_freq_2000_supp_0.3_w_ATE.pkl'
    elif dataset_name == "Stackoverflow Annual Developer Survey":
        rules_dir = f'data/so/rules_freq_1000_supp_0.3.pkl'
    elif dataset_name == "Compas":
        pass

    with open(rules_dir, 'rb') as f:
        temp = pickle.load(f)
    
    _rules = [(rule, support, del_ATE) for rule, support, del_ATE in temp.values()]
    _rules_temp = [(rule_to_predicate_cg(rule), support, del_ATE) for rule, support, del_ATE in temp.values()]
    rules = pd.DataFrame(data=_rules_temp, columns=['Rules', 'Support', 'Δ ATE'])
    # rules['Δ ATE'] = np.random.uniform(5, 10, len(rules))

    st.session_state["_rules"] = _rules
    st.session_state["rules"] = rules    

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
        st.write("📂 Select a Dataset and Load the Training and Test Data.")

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

            from .util import custom_css, cell_style_jscode, cell_style_jscode2

            gb.configure_column("income", headerClass="prediction-header", cellStyle=cell_style_jscode)
            # gb.configure_column("income", headerClass="truevalue-header", cellStyle=cell_style_jscode2)

            grid_options = gb.build()

            # Create narrow column to shrink the grid
            col1, col2 = st.columns([0.9, 0.1])  # Adjust column width ratio

            with col1:
                # st.subheader("Select Test Row:")
                grid_response = AgGrid(
                    data,
                    gridOptions=grid_options,
                    fit_columns_on_grid_load=True,
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

            from .util import custom_css, cell_style_jscode, cell_style_jscode2

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
                    fit_columns_on_grid_load=True,
                    height=250,
                    enable_enterprise_modules=False,
                    allow_unsafe_jscode=True,
                    custom_css=custom_css
                )

    elif st.session_state.step == 2:
        st.write("⚙️ Choose a machine learning model and run classification.")

        if st.button("🚀 Train Model"):
            with st.spinner(f'Training {st.session_state.selected_model} Model...'):
                if st.session_state.selected_model == "NN":
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

                from .util import custom_css, cell_style_jscode, cell_style_jscode2

                gb.configure_column("income (Prediction)", headerClass="prediction-header", cellStyle=cell_style_jscode)
                gb.configure_column("income (True Value)", headerClass="truevalue-header", cellStyle=cell_style_jscode2)

                grid_options = gb.build()

                # Create narrow column to shrink the grid
                col1, col2 = st.columns([0.9, 0.1])  # Adjust column width ratio

                with col1:
                    st.subheader("Select Test Row:")
                    grid_response = AgGrid(
                        data,
                        gridOptions=grid_options,
                        fit_columns_on_grid_load=True,
                        height=250,
                        enable_enterprise_modules=False,
                        allow_unsafe_jscode=True,
                        custom_css=custom_css
                    )
                st.success("✅ Classification results generated!")


    elif st.session_state.step == 3:
        st.write("⚙️ **Upload Causal DAG ➡ Set Parameters ➡ Generate Profiles**")
        st.write("Please provide a causal graph in the form of a directed acyclic graph (DAG) in the .txt format. The graph should be specified as a list of edges, where each edge is represented by a pair of nodes separated by a space. For example, an edge from node A to node B would be represented as 'A B'.") 
                 
        st.write("Profiles are sets of of features causally related to the target variable expressed in disjunctive normal form, for example, `((A1 ∧ A2) ∨ (B1 ∧ B2))`. The profiles will be generated based on the provided causal graph and the specified parameters.")

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
        st.write("Rules are predicates that describe the subsets in the training data. Explanation sets are also described by rules. Explanation sets with small sizes are interesting because they are easier to explain. Also, removing subsets of the training data affects the predictions for rows other than the intended row. Smaller explanation sets minimize the side effects. Support is the fraction of the explanation set size to the full training data. Max support allows to filter out explanation sets that are too large.") 
        st.write("The system also uses the change in average treatment effect to identify causally important rules. The Min Δ ATE is a threshold to identify important rules. Rules with Δ ATE below the threshold are filtered out.")

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

                gb = GridOptionsBuilder.from_dataframe(rules)
                gb.configure_selection(selection_mode="single")  # Select entire row on click
                # gb.configure_pagination(paginationAutoPageSize=True)  # Enable pagination
                # gb.configure_column("Index", pinned="left")  # Pin index column to the left
                # gb.configure_column('Rules', width=100)
                gb.configure_column('Rules', width=1100, auto_size=True, wrapText=True, autoHeight=True)  # Automatically adjust width
                gb.configure_column('Support', flex=1, valueFormatter="(value != null) ? value.toFixed(3) : ''" )  # Automatically adjust width
                gb.configure_column('Δ ATE', flex=1, valueFormatter="(value != null) ? value.toFixed(3) : ''" )  # Automatically adjust width
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

        # Single Button in Step 5
        if st.button("✅ Confirm and Proceed"):
            with st.spinner(f'Retraining...'):
                time.sleep(5)
            st.success("✅ Retraining finished!")  # Placeholder for future logic

    # Main Content for Step 6
    elif st.session_state.step == 6:
        # st.subheader("⚙️ Running the analysis... Please wait.")
        st.write("⚙️ **Select a Test row ➡ Select Number of Explanation Sets ➡ Press Generate Explanation Sets ➡ Select Explanation**")

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

            from .util import custom_css, cell_style_jscode, cell_style_jscode2


            gb.configure_column("income (Prediction)", headerClass="prediction-header", cellStyle=cell_style_jscode)
            gb.configure_column("income (True Value)", headerClass="truevalue-header", cellStyle=cell_style_jscode2)

            grid_options = gb.build()

            # Create narrow column to shrink the grid

            st.subheader("Select Test Row:")
            grid_response = AgGrid(
                data,
                gridOptions=grid_options,
                fit_columns_on_grid_load=True,
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
                profiles = st.session_state._profiles

                selected_index = test_indices[int(selected_rows.index[0])]
                row = selected_rows.iloc[[0]]

                _profiles = [profiles[1315], profiles[1801], profiles[3040]]
                
                st.subheader("Explanation Rules:")

                # explanations = [' or '.join(['(' + ' and '.join([f'`{c}`=="{row.iloc[0][c]}"' for c in conjunction + ')']) for conjunction in profile])for profile in _profiles]
                explanations = [' or '.join(['(' + ' and '.join([f'`{c}`=="{row.iloc[0][c]}"' for c in conjunction]) + ')' for conjunction in profile]) for profile in _profiles]
                
                for explanation in explanations:
                    if st.button(explanation):
                        if explanation == explanations[0]:
                            profile = profiles[1315]
                            nrows = 572
                            suport = nrows/len(st.session_state.training_data)
                            delta_ATE = 6.387
                            accuracy = 0.815
                            model_similarity = 0.971
                            

                        elif explanation == explanations[1]:
                            profile = profiles[1801]
                            nrows = 941
                            suport = nrows/len(st.session_state.training_data)
                            delta_ATE = 5.141
                            accuracy = 0.809
                            model_similarity = 0.973
                            
                        elif explanation == explanations[2]:
                            profile = profiles[3040]
                            nrows = 858
                            suport = nrows/len(st.session_state.training_data)
                            delta_ATE = 6.032
                            accuracy = 0.811
                            model_similarity = 0.966
                            

                        # row = st.session_state['selected_row']
                        predicate = ' or '.join([' and '.join([f'`{c}`=="{row.iloc[0][c]}"' for c in conjunction]) for conjunction in profile])

                        # st.write(f"{explanation}: {predicate}")

                        train_df = st.session_state.training_data
                        exp_set = train_df.query(predicate)
                    
                        
                        col1, col2 = st.columns([0.6, 0.4])

                        with col1:
                            st.subheader(f"Explanation Set: {nrows} rows, Support {suport:.3f}, Δ ATE: {delta_ATE:.3f}, Accuracy: {accuracy:.3f}, Model Similarity: {model_similarity:.3f}")

                            st.dataframe(exp_set, height=200)

                        with col2:
                            st.subheader("Causal Paths Supported by Explanation")
                            import pygraphviz as pgv
                            import io
                            A = highlighted_graph(profile)
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