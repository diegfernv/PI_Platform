import re
from typing import List, Optional

import gradio as gr
import pandas as pd

from preprocessing.embeddings import ProteinEmbedder
from preprocessing.encoders import OneHotEncoder, OrdinalEncoder, KMersEncoder, FrequencyEncoder, PhysicochemicalEncoder, FFTEncoder
from utils import load_csv, get_gpu_devices, get_properties_names, find_matching_column, generate_preview
from constants import ENCODING_TYPES, EMBEDDING_MODELS

# TODO delete [:50] in datasets
# TODO Cambiar a un fft a un checkbox de "apply fft"

# NOTE: Probablemente tenga que anadir el parametro response_col en el
# resto, ya que en embedding se utiliza para generar los graficos, cosa que no hace one hot

### Section: Data Preprocessing ###

def use_embedding(
        file,
        sequence_col: str,
        response_col: str,
        device: str,
        model: str,
        max_length: int,
        batch_size: int = 10):
    """Use a pre-trained model to encode sequences."""
    try:
        instance = ProteinEmbedder(
            device=device, 
            model_name=model,
            dataset=file[:50], 
            column_seq=sequence_col, 
            column_label=response_col,
            columns_ignore=[response_col]
        )
        if "esmc" in instance.model_name:
            instance.loadModel()
        else:
            instance.loadModelTokenizer()
        
        instance.embeddingProcess(batch_size=batch_size)
        return instance.getDataFrame()
    except Exception as e:
        raise gr.Error(f"Error in embedding process: {str(e)}")
    
def use_one_hot(
        file,
        sequence_col: str,
        ignore_columns: List[str],
        max_length: int):
    """Use one-hot encoding for sequences."""
    try:
        instance = OneHotEncoder(
            dataset=file[:50], 
            sequence_column=sequence_col, 
            ignore_columns=ignore_columns,
            max_length=max_length
        )
        instance.run_process()
        return instance.coded_dataset
    except Exception as e:
        raise gr.Error(f"Error in one-hot encoding process: {str(e)}")

def use_ordinal(
        file,
        sequence_col: str,
        ignore_columns: List[str],
        max_length: int):
    """Use ordinal encoding for sequences."""
    try:
        instance = OrdinalEncoder(
            dataset=file[:50],
            sequence_column=sequence_col,
            ignore_columns=ignore_columns,
            max_length=max_length
        )
        instance.run_process()
        return instance.coded_dataset
    except Exception as e:
        raise gr.Error(f"Error in ordinal encoding process: {str(e)}")

def use_kmers(
        file,
        sequence_col: str,
        ignore_columns: List[str],
        size_kmer: int):
    """Use k-mer encoding for sequences."""
    try:
        instance = KMersEncoder(
            dataset=file[:50],
            sequence_column=sequence_col,
            ignore_columns=ignore_columns,
            size_kmer=size_kmer
        )
        instance.run_process()
        return instance.coded_dataset
    except Exception as e:
        raise gr.Error(f"Error in k-mer encoding process: {str(e)}")
    
def use_frequency(
        file,
        sequence_col: str,
        ignore_columns: List[str],
        max_length: int):
    """Use frequency encoding for sequences."""
    try:
        instance = FrequencyEncoder(
            dataset=file[:50],
            sequence_column=sequence_col,
            ignore_columns=ignore_columns,
            max_length=max_length
        )
        instance.run_process()
        return instance.coded_dataset
    except Exception as e:
        raise gr.Error(f"Error in frequency encoding process: {str(e)}")

def use_physicochemical(
        file,
        sequence_col: str,
        ignore_columns: List[str],
        max_length: int,
        name_property: str):
    """Use physicochemical properties for encoding sequences."""
    try:
        aaindex_encoder_df = load_csv("../input_config/aaindex_encoders.csv")
        aaindex_encoder_df.index = aaindex_encoder_df["residue"].values
        instance = PhysicochemicalEncoder(
            dataset=file[:50],
            sequence_column=sequence_col,
            ignore_columns=ignore_columns,
            max_length=max_length,
            name_property=name_property,
            df_properties=aaindex_encoder_df
        )
        instance.run_process()
        return instance.df_data_encoded
    except Exception as e:
        raise gr.Error(f"Error in physicochemical encoding process: {str(e)}")

def encode_data(
        file, 
        sequence_col: str, 
        response_col: str,
        ignore_cols: List[str],
        device: str, 
        encoding: str, 
        max_length: Optional[int], 
        name_property: Optional[str], 
        size_kmer: Optional[int], 
        embedding_model: Optional[str],
        batch_size: Optional[int]):
    """Process the input dataframe and return the result."""
    try:
        df = load_csv(file.name)
        
        if sequence_col not in df.columns or response_col not in df.columns:
            raise ValueError("Selected columns are not in the dataframe.")
        
        device = get_gpu_devices().get(device, "cpu")

        match encoding:
            case "One-Hot":
                encoded = use_one_hot(df, sequence_col, ignore_cols, max_length)
            case "Ordinal":
                encoded = use_ordinal(df, sequence_col, ignore_cols, max_length)
            case "Frequency":
                encoded = use_frequency(df, sequence_col, ignore_cols, max_length)
            case "KMer":
                encoded = use_kmers(df, sequence_col, ignore_cols, size_kmer)
            case "Physicochemical":
                encoded = use_physicochemical(df, sequence_col, ignore_cols, max_length, name_property)
            case "Embedding":
                model = EMBEDDING_MODELS[embedding_model]
                encoded = use_embedding(df, sequence_col, response_col, device, model, max_length, batch_size)
            
    except Exception as e:
        raise gr.Error(f"Error processing data: {str(e)}")
    
    # Generate preview of the encoded data
    encoded = generate_preview(df, encoded, sequence_col, response_col)
    
    return gr.update(
        value=encoded.head(),
        headers=encoded.columns.tolist(),
        row_count=min(10, len(encoded)),
        col_count=(len(encoded.columns), "fixed"),
        interactive=False,
        visible=True
    )

def update_on_upload(file):
    """Update dropdowns and preview based on the uploaded file."""
    # Load partially to get column names
    try:
        df = pd.read_csv(file.name, nrows=5)
    except Exception as e:
        return gr.Error(f"Error loading file: {str(e)}")
    
    show_ignore = False
    
    columns = df.columns.tolist()
    sequence_col = find_matching_column(columns, ["sequence", "seq"])
    response_col = find_matching_column(columns, ["response", "label"])

    if len(columns) > 2:
        show_ignore = True

    return gr.update(
        choices=columns,
        value=sequence_col,
        visible=True
    ), gr.update(
        choices=columns,
        value=response_col,
        visible=True
    ), gr.update(
        choices=[col for col in columns if col not in [sequence_col, response_col]],
        value=[],
        visible=show_ignore
    ), gr.update(
        value=df,
        visible=True
    )

def update_encoding_parameters(encoding_type):
    """Update encoding parameters based on the selected encoding type."""
    max_length_input = gr.update(visible=False)
    name_property_input = gr.update(visible=False)
    size_kmer_input = gr.update(visible=False)
    model_selector = gr.update(visible=False)
    batch_size_input = gr.update(visible=False)

    if encoding_type in ["One-Hot", "Ordinal", "Frequency"]:
        max_length_input = gr.update(visible=True)
    elif encoding_type == "KMer":
        size_kmer_input = gr.update(visible=True)
    elif encoding_type in ["Physicochemical", "FFT"]:
        name_property_input = gr.update(
            choices=get_properties_names(),
            value=get_properties_names()[0],
            visible=True
        )
        max_length_input = gr.update(visible=True)
    elif encoding_type == "Embedding":
        model_selector = gr.update(
            choices=list(EMBEDDING_MODELS.keys()),
            value=list(EMBEDDING_MODELS.keys())[0],
            visible=True
        )
        batch_size_input = gr.update(
            value=10,
            visible=True
        )
    
    return max_length_input, name_property_input, size_kmer_input, model_selector, batch_size_input

    

def init_preprocessing():
    """Initialize the data preprocessing interface."""
    gpu_devices = get_gpu_devices()

    ## Interface ##

    with gr.Blocks() as preprocessing:
        gr.Markdown("## Data Preprocessing")
        gr.Markdown("Upload a CSV file to display its contents.")
        with gr.Row():
            
            file_input = gr.File(
                label="Upload CSV File", 
                file_types=[".csv"]
            )
        
        gr.Markdown("### Data Preview")
        
        preview = gr.Dataframe(
            visible=False,
            interactive=False
        )

        # TODO remove option selected in one dropdown from the other dropdown
        with gr.Row():
            dropdown_sequence_col = gr.Dropdown(
                label="Sequence Column",
                interactive=True,
                info="Select the column containing sequences",
                visible=False
            )
            dropdown_response_col = gr.Dropdown(
                label="Response Column",
                interactive=True,
                info="Select the column containing responses",
                visible=False
            )
            checkbock_group_ignore = gr.CheckboxGroup(
                label="Ignore Columns",
                choices=[],
                interactive=True,
                info="Select columns to ignore",
                visible=False
            )
        
        with gr.Row():
            checkbox_cuda = gr.Checkbox(
                label="Use CUDA",
                value=True if gpu_devices else False,
                info="Use GPU acceleration if available.",
                interactive=True if gpu_devices else False
            )
            device_selector = gr.Dropdown(
                label="Device",
                choices=list(gpu_devices.keys()),
                value=list(gpu_devices.keys())[0] if gpu_devices else "cpu",
                info="Select the device for processing.",
                interactive=True
            )
        
        with gr.Row():
            encoding_selector = gr.Dropdown(
                label="Select Model",
                choices=ENCODING_TYPES,
                value=ENCODING_TYPES[0],
                info="Select the model for processing.",
                interactive=True
            )
            max_length_input = gr.Number(
                label="Max Length",
                value=50,
                info="Maximum length for encoding.",
                interactive=True,
                visible=False
            )
            name_property_input = gr.Dropdown(
                label="Select Property",
                info="Select the property for encoding.",
                interactive=True,
                visible=False
            )
            size_kmer_input = gr.Number(
                label="K-mer Size",
                value=3,
                info="Size of the k-mer for encoding.",
                interactive=True,
                visible=False
            )
            model_selector = gr.Dropdown(
                label="Select Model",
                info="Select the model for encoding.",
                interactive=True,
                visible=False
            )
            batch_size_input = gr.Number(
                label="Batch Size",
                info="Batch size for processing.",
                interactive=True,
                visible=False
            )


        with gr.Row():
            process_btn = gr.Button("Process Data", variant="primary", interactive=True)
                
    
        gr.Markdown("### Result Preview")
        
        with gr.Row():
            result_preview = gr.Dataframe(
                visible=False,
                interactive=False
            )

        ## Logic ##

        file_input.upload(
            fn=update_on_upload,
            inputs=file_input,
            outputs=[dropdown_sequence_col, dropdown_response_col, checkbock_group_ignore, preview]
        )

        checkbox_cuda.change(
            fn=lambda x: gr.update(visible=x),
            inputs=checkbox_cuda,
            outputs=device_selector
        )

        encoding_selector.change(
            fn=update_encoding_parameters,
            inputs=encoding_selector,
            outputs=[
                max_length_input,
                name_property_input,
                size_kmer_input,
                model_selector,
                batch_size_input
            ]
        )
        process_btn.click(
            fn=encode_data,
            inputs=[
                file_input,
                dropdown_sequence_col,
                dropdown_response_col,
                checkbock_group_ignore,
                device_selector,
                encoding_selector,
                max_length_input,
                name_property_input,
                size_kmer_input,
                model_selector,
                batch_size_input
            ],
            outputs=[
                result_preview
            ]
        )
       
    return preprocessing