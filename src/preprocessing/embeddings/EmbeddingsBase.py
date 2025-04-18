from typing import List
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from logger import get_logger
from time import sleep

class EmbeddingsBase(object):

    def __init__(
            self,
            name_device: str = "cuda",
            name_model: str = None,
            name_tokenizer: str = None,
            dataset: pd.DataFrame = None,
            column_seq: str = None,
            columns_ignore: List[str] = []):
        
        self.name_device = name_device
        self.name_model = name_model
        self.name_tokenizer = name_tokenizer
        self.dataset = dataset
        self.column_seq = column_seq
        self.columns_ignore = columns_ignore

        self.tokenizer = None
        self.model = None

        self.embeddings = []

        self.status = True
        self.message = ""

        self.logger = get_logger(f"EmbeddingsBase({self.name_model})")

        self.__select_device()

    def __select_device(self):
        self.device = torch.device(self.name_device)
        self.logger.info(f"Using device: {self.device}")
    
    def loadModelTokenizer(self):
        pass

    def cleaning_memory(self):
        self.logger.info("Cleaning cache memory")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.logger.info("Current memory usage: {:.2f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))

    
    def embeddingBatch(self, batch, max_length: int = 1024):
        
        inputs = self.tokenizer(
            batch, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            add_special_tokens=False, 
            max_length=max_length).to(self.device)
        
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            all_hiden_layers = self.model(**inputs, output_hidden_states=False)
            
        del inputs

        return all_hiden_layers.last_hidden_state
        
    def embeddingProcess(self, batch_size: int = 100):

        sequences = self.dataset[self.column_seq].tolist()

        layer_embeddings = []

        for i in tqdm(range(0, len(sequences), batch_size), desc="[+] Embedding", unit="batch"):
            batch = sequences[i:i + batch_size]

            last_hidden_layer = self.embeddingBatch(batch=batch)

            batch_embedding = last_hidden_layer.mean(dim=1).cpu().numpy()
            layer_embeddings.append(batch_embedding)

        layer_embeddings = np.concatenate(layer_embeddings, axis=0)
        
        header = [f"p_{i+1}" for i in range(layer_embeddings.shape[1])]

        df_embedding = pd.DataFrame(data=layer_embeddings, columns=header)
        
        for column in self.columns_ignore:
            df_embedding[column] = self.dataset[column].values
        
        return df_embedding
