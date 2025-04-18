from typing import List, Optional
import re

import numpy as np
import pandas as pd

import torch
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

def applySplit(
        dataset: np.ndarray,
        response: np.ndarray = None, 
        test_size: float = 0.1, 
        val_size: float = 0.2, 
        random_state: int = 42
        ):
    train_val_X, test_X, train_val_y, test_y = train_test_split(dataset, response, test_size=test_size, random_state=random_state)
    train_X, val_X, train_y, val_y = train_test_split(train_val_X, train_val_y, test_size=val_size, random_state=random_state)
    return train_X, val_X, test_X, train_y, val_y, test_y


def makeScoresForRegression():
    scoring = {
        'MAE': make_scorer(mean_absolute_error),
        'MSE': make_scorer(mean_squared_error),
        'R2': make_scorer(r2_score),
        'RMSE' : make_scorer(root_mean_squared_error)
    }

    return scoring

def scalerProcess(dataset):
    scaler = MaxAbsScaler()
    scaler.fit(dataset)
    dataset_scaled = scaler.transform(dataset)

    dataset_scaled = pd.DataFrame(data=dataset_scaled, columns=dataset.columns)
    return dataset_scaled, scaler

def is_aa_sequence(sequence):
    alphabet = set("ACDEFGHIKLMNPQRSTVWY")
    sequence = sequence.strip()
    is_canon = True
    for res in set(sequence):
        if res not in alphabet:
            is_canon = False
    return is_canon

### Interface tools ###

def find_matching_column(columns: List[str], patterns: List[str]) -> str:
    """Auto select columns from a dataframe based on patterns."""
    for pattern in patterns:
        for col in columns:
            if re.search(pattern.lower(), col, re.IGNORECASE):
                return col
    return columns[0] if columns else None

def load_csv(filename: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {str(e)}")
    
def get_gpu_devices() -> dict:
    gpu_devices = {}

    if torch.cuda.is_available():
        device_cound = torch.cuda.device_count()
        for i in range(device_cound):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_devices.update({gpu_name: f"cuda:{i}"})
    else:
        gpu_devices.update({"CPU": "cpu"})
    
    return gpu_devices
            
def get_properties_names() -> List[str]:
    """Get the names of the properties."""
    df = pd.read_csv("../input_config/aaindex_encoders.csv", nrows=0)
    properties = df.columns.tolist()[1:]
    return properties


def generate_preview(df: pd.DataFrame, encoded_data: pd.DataFrame, sequence_column: str, response_column: Optional[str]) -> pd.DataFrame:
    """Generate a preview of the dataframe with reduced number of columns."""
    df = df.copy()
    # Get the number of columns that have p_1 to p_n
    
    # Get the first 3 columns (p_1, p_2, p_3) and the last 3 columns (p_n-2, p_n-1, p_n)
    first_columns = encoded_data.iloc[:, :3]
    last_columns = encoded_data.iloc[:, -3:]

    # Concatenate the first and last columns with the sequence and response columns
    df = pd.concat([
        df[sequence_column], 
        df[response_column] if response_column else pd.Series(),
        first_columns, 
        last_columns
        ], axis=1)
    df.insert(len(df.columns)-3,"...", "...")

    return df