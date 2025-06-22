import streamlit as st
from demo_ui.sidebar import render_sidebar
from demo_ui.initialize import initialize_session_variables
from demo_ui.main_pane import render_main_pane


def main():
    st.set_page_config(layout="wide")

    initialize_session_variables()

    st.session_state.time_now += 1
    # st.write("Time: ", st.session_state.time_now)

    datasets = ["Adult Income", "Stackoverflow Annual Developer Survey", "Compas"]

    ml_models = ["SVM", "Logistic Regression", "Neural Network", "Adaboost", "Random Forest"]

    steps = {
        1: "Select Dataset",
        2: "Select Model",
        3: "Set Causal DAG & Generate Profiles",
        4: "Filter Rules",
        5: "Retrain",
        6: "Generate Explanation Set",
    }

    render_sidebar(steps, datasets, ml_models)

    render_main_pane(steps)

   
if __name__ == "__main__":
    main()
