import gradio as gr
import pandas as pd

### Section: Data Retrieval ###

def init_data_retrieval():
    return gr.Interface(
        fn=lambda x, y: pd.DataFrame({"name": x, "age": y}),
        inputs=[
            gr.Dataframe(
                headers=["name", "age", "gender"],
                datatype=["str", "number", "str"],
                row_count=5,
                col_count=(3, "fixed"),
            ),
            gr.Dropdown(["M", "F", "O"]),
        ],
        outputs="dataframe",
        description="Enter gender as 'M', 'F', or 'O' for other.",
    )