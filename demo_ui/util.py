from st_aggrid import JsCode

custom_css = {
    ".prediction-header": {
        # "background-color": "red !important",
        # "color": "white !important",
        "font-weight": "bold !important",
        "text-align": "center",
        'border': '1px solid pink'
    },
    ".truevalue-header": {
        # "background-color": "red !important",
        # "color": "white !important",
        "font-weight": "bold !important",
        "text-align": "center",
        'border': '1px solid lightgreen'
    }

}

cell_style_jscode = JsCode("""
    function(params) {
        if (params.data && params.data["income (Prediction)"] !== params.data["income (True Value)"]) {
            return { backgroundColor: "pink", 'border': '1px solid pink'};
        }
        return {'border': '1px solid pink'};
    }   
""")

cell_style_jscode2 = JsCode("""
    function(params) {
        if (params.data && params.data["income (Prediction)"] !== params.data["income (True Value)"]) {
            return { backgroundColor: "lightgreen", 'border': '2px solid lightgreen'};
        }
        return {'border': '1px solid lightgreen'};
    }   
""")

def make_cell_style_jscode(target):
    return JsCode(f"""
        function(params) {{
            if (params.data && params.data["{target} (Prediction)"] !== params.data["{target} (True Value)"]) {{
                return {{ backgroundColor: "pink", 'border': '3px solid pink'}};
            }}
            return {{'border': '3px solid pink'}};
        }}   
    """)    

def make_cell_style_jscode2(target):
    return JsCode(f"""
        function(params) {{
            if (params.data && params.data["{target} (Prediction)"] !== params.data["{target} (True Value)"]) {{
                return {{ backgroundColor: "lightgreen", 'border': '3px solid lightgreen'}};
            }}
            return {{'border': '3px solid lightgreen'}};
        }}   
    """)