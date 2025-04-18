import re
from typing import List

import torch
import gradio as gr
import pandas as pd

from logger import get_logger
from preprocessing.embeddings import ProteinEmbedder
from interface.data_retrieval import init_data_retrieval
from interface.preprocessing import init_preprocessing
from interface.ml_tools import init_ml_tools


from constants import ENCODING_TYPES, EMBEDDING_MODELS

### Section: Main Function ###

if __name__ == "__main__":
    data_retrieving = init_data_retrieval()
    preprocessing = init_preprocessing()
    ml_tools = init_ml_tools()

    demo = gr.TabbedInterface(
        [data_retrieving, preprocessing, ml_tools],
        ["Data Retrieving", "Preprocessing", "Machine Learning Tools"]
    )

    demo.launch()