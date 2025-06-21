import streamlit as st
import networkx as nx


def store_value(key):
    st.session_state[key] = st.session_state["_"+key]

def load_value(key):
    st.session_state["_"+key] = st.session_state[key]


def render_sidebar(steps, datasets, ml_models):
    """
    """

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            min-width: 250px;  /* Adjust this value */
            max-width: 250px;  /* Adjust this value */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Sidebar - Updates dynamically per step
    with st.sidebar:
        st.header(f"Step {st.session_state.step}: {steps[st.session_state.step]}")

        if st.session_state.step == 1:
            # uploaded_file1 = st.file_uploader("Upload Training Data (.csv)", type=["csv"])
            # uploaded_file2 = st.file_uploader("Upload Test Data (.csv)", type=["csv"])
            load_value("selected_dataset")
            st.selectbox("Select a Dataset:", datasets, key="_selected_dataset", on_change=store_value, args=["selected_dataset"])

        elif st.session_state.step == 2:
            load_value("selected_model")
            st.selectbox("Select a Black-box ML Algorithm:", ml_models, key="_selected_model", on_change=store_value, args=["selected_model"])

        elif st.session_state.step == 3:
            # uploaded_file3 = st.file_uploader("Upload Causal DAG (.txt)", type=["txt"])
            # if uploaded_file3:
            #     graph_data = uploaded_file3.read().decode("utf-8")
            #     G = nx.parse_edgelist(graph_data.splitlines(), create_using=nx.DiGraph)
            #     st.session_state['CG'] = G

            st.write("🔧 **Set Profile Parameters**")
            load_value("conj_max_len")
            st.number_input(label="Conjunction max length", min_value=1, max_value=10, step=1, key="_conj_max_len", on_change=store_value, args=["conj_max_len"])
            
            load_value("disj_max_len")
            st.number_input(label="Disjunction max length", min_value=1, max_value=10, step=1, key="_disj_max_len", on_change=store_value, args=["disj_max_len"])

        elif st.session_state.step == 4:
            st.write("⚙️ **Set Rule Parameters**")

            load_value("support_thrs")
            st.slider(label="Max Support", min_value=0.0, max_value=1.0, step=0.05, key="_support_thrs", on_change=store_value, args=["support_thrs"])
            
            load_value("delta_ATE")
            st.number_input(label="Min Δ ATE", min_value=0.0, step=0.1, key="_delta_ATE", on_change=store_value, args=["delta_ATE"])
        
        elif st.session_state.step == 5:  # Show read-only parameters in Step 5
            st.write("📜 **Review Model Selection and Parameters**")
            
            load_value("selected_model")
            st.selectbox("Black-box ML Algorithm", ml_models, key="_selected_model", on_change=store_value, args=["selected_model"], disabled=True)

            # Model Parameters (Normal inputs)
            load_value("conj_max_len")
            st.number_input("Conjunction Max Length", min_value=1, max_value=10, step=1, key="_conj_max_len", on_change=store_value, args=["conj_max_len"], disabled=True)
            
            load_value("disj_max_len")
            st.number_input("Disjunction Max Length", min_value=1, max_value=10, step=1, key="_disj_max_len", on_change=store_value, args=["disj_max_len"], disabled=True)
            
            load_value("support_thrs")
            st.slider("Support", 0.0, 1.0, step=0.05, key="_support_thrs", on_change=store_value, args=["support_thrs"], disabled=True)
            
            load_value("delta_ATE")
            st.number_input("Δ ATE", min_value=0.0, step=0.1, key="_delta_ATE", on_change=store_value, args=["delta_ATE"], disabled=True)

        elif st.session_state.step == 6:  # Show read-only parameters in Step 5
            st.write("**Model Selection and Parameters**")
            # Model Selection (Dropdown)
            load_value("selected_model")
            st.selectbox("Black-box ML Algorithm", ml_models, key="_selected_model", on_change=store_value, args=["selected_model"], disabled=True)

            # Model Parameters (Normal inputs)
            load_value("conj_max_len")
            st.number_input("Conjunction Max Length", min_value=1, max_value=10, step=1, key="_conj_max_len", on_change=store_value, args=["conj_max_len"], disabled=True)
            
            load_value("disj_max_len")
            st.number_input("Disjunction Max Length", min_value=1, max_value=10, step=1, key="_disj_max_len", on_change=store_value, args=["disj_max_len"], disabled=True)
            
            load_value("support_thrs")
            st.slider("Support", 0.0, 1.0, step=0.05, key="_support_thrs", on_change=store_value, args=["support_thrs"], disabled=True)
            
            load_value("delta_ATE")
            st.number_input("Δ ATE", min_value=0.0, step=0.1, key="_delta_ATE", on_change=store_value, args=["delta_ATE"], disabled=True)
            
            load_value("num_exp_set")
            st.number_input("Number of explanation sets", min_value=1, step=1, key="_num_exp_set", on_change=store_value, args=["num_exp_set"])

            # Button to generate explanation sets
            if st.button("📝 Generate Explanation Set"):
                if st.session_state.step6_selected_row is None:
                    st.write("⚠️ **Select a test row first.**")
                elif st.session_state.step6_exp_set_button == 1:
                    st.session_state.step6_exp_set_button = 0
                    st.success("✅ Explanation Sets Generated!")
                    
