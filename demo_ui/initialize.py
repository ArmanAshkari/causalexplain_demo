import streamlit as st
import pandas as pd

def initialize_session_variables():
    """
    """
    # Initialize session state
    if "time_now" not in st.session_state:
        st.session_state.time_now = 0

    if "step" not in st.session_state:
        st.session_state.step = 1

    if "selected_dataset" not in st.session_state:
        st.session_state.selected_dataset = "Adult Income"
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "SVM"
    
    # if "training_data" not in st.session_state:
    #     st.session_state.training_data = None
    
    # if "test_data" not in st.session_state:
    #     st.session_state.test_data = None
    
    if "classification_results" not in st.session_state:
        st.session_state.classification_results = None
    
    if "causal_graph" not in st.session_state:
        st.session_state.causal_graph = None
    
    if "profiles" not in st.session_state:
        st.session_state.profiles = pd.DataFrame(columns=["Profiles"])
    
    if "rules" not in st.session_state:
        st.session_state.rules = pd.DataFrame(columns=["Rules", "Δ ATE"])
    
    if "selected_profile" not in st.session_state:
        st.session_state.selected_profile = None
    
    if "conj_max_len" not in st.session_state:
        st.session_state.conj_max_len = 3
    
    if "disj_max_len" not in st.session_state:
        st.session_state.disj_max_len = 2
    
    if "num_exp_set" not in st.session_state:
        st.session_state.num_exp_set = 2
    
    if "support_thrs" not in st.session_state:
        st.session_state.support_thrs = 0.2
    
    if "delta_ATE" not in st.session_state:
        st.session_state.delta_ATE = 4.0
    
    if "table_data" not in st.session_state:
        st.session_state.table_data = pd.DataFrame(columns=["Feature", "Value"])  # Initialize as empty DataFrame
    
    if "df1" not in st.session_state:
        st.session_state.df1 = pd.DataFrame({"Feature A": [1, 2, 3], "Feature B": [4, 5, 6]})
    
    if "df2" not in st.session_state:
        st.session_state.df2 = pd.DataFrame({"Metric X": [10, 20, 30], "Metric Y": [40, 50, 60]})
    
    if "df3" not in st.session_state:
        st.session_state.df3 = pd.DataFrame({"Parameter M": [100, 200, 300], "Parameter N": [400, 500, 600]})
    
    if "step6_image" not in st.session_state:
        st.session_state.step6_image = "https://upload.wikimedia.org/wikipedia/commons/8/88/GraphCycle.svg"
    
    # Initialize session state for row selection tracking
    if "step3_selected_row" not in st.session_state:
        st.session_state.step3_selected_row = None

    if "step4_selected_row" not in st.session_state:
        st.session_state.step4_selected_row = None
    
    if "step6_selected_row" not in st.session_state:
        st.session_state.step6_selected_row = None

    if "step6_exp_set_button" not in st.session_state:
        st.session_state.step6_exp_set_button = 0

