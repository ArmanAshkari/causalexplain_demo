import streamlit as st
import pandas as pd
import numpy as np

# Generate sample DataFrame
np.random.seed(42)
# data = pd.DataFrame(np.random.randint(0, 100, size=(10, 20)), columns=[f'Col_{i}' for i in range(1, 21)])

# # Introduce some mismatches in the second last column
# data.iloc[2, -2] = data.iloc[2, -1] + 5  # Mismatch example
# data.iloc[5, -2] = data.iloc[5, -1] - 3  # Another mismatch

# Function to highlight only mismatched cells in pink
def highlight_mismatch(val, reference):
    return 'background-color: pink' if val != reference else ''

def apply_highlight(df, target):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    for i in df.index:
        if df.loc[i, f'{target} (Prediction)'] != df.loc[i, f'{target} (True Value)']:  # Check mismatch
            # styles.loc[i, f'{target} (Prediction)'] = 'background-color: pink;'
            # styles.loc[i, f'{target} (True Value)'] = 'background-color: lightgreen;'
        # styles.loc[i, f'{target} (Prediction)'] += 'border: 2px solid pink;'
        # styles.loc[i, f'{target} (True Value)'] += 'border: 2px solid lightgreen;'
            styles.loc[i, f'{target} (Prediction)'] = 'border: 3px solid pink;'
            styles.loc[i, f'{target} (True Value)'] = 'border: 3px solid lightgreen;'
    
    # # Modify the existing style object to add borders
    # styles = styles.set_table_styles(
    #     [
    #         {'selector': f'th:nth-child({df.columns.get_loc(f"{target} (Prediction)") + 1})', 'props': [('border', '1px solid pink')]} 
    #     ] + [
    #         {'selector': f'td:nth-child({df.columns.get_loc(f"{target} (Prediction)") + 1})', 'props': [('border', '1px solid pink')]} 
    #     ] + [
    #         {'selector': f'th:nth-child({df.columns.get_loc(f"{target} (True Value)") + 1})', 'props': [('border', '1px solid lightgreen')]} 
    #     ] + [
    #         {'selector': f'td:nth-child({df.columns.get_loc(f"{target} (True Value)") + 1})', 'props': [('border', '1px solid lightgreen')]} 
    #     ],
    #     overwrite=False  # Keep existing styles and add new ones
    # )
    return styles

# # Apply styling function
# styled_df = data.style.apply(apply_highlight, axis=None)

# # Streamlit app
# st.title("DataFrame with Highlighted Mismatched Cells")
# st.dataframe(styled_df)

