from typing import List
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd

from .EmbeddingsBase import EmbeddingsBase

class MistralBased(EmbeddingsBase):

    def __init__(
            self, 
            name_device:str = "cuda", 
            name_model: str = "RaphaelMourad/Mistral-Prot-v1-134M", 
            name_tokenizer: str = "RaphaelMourad/Mistral-Prot-v1-134M",
            dataset: str = None,
            column_seq: str = None,
            columns_ignore: List[str] = []):
        
        super().__init__(name_device, name_model, name_tokenizer, dataset, column_seq, columns_ignore)

    def loadModelTokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.name_model, 
            trust_remote_code=True)
        
        self.model = AutoModel.from_pretrained(
            self.name_model,
            trust_remote_code=True).to(self.device)
        
        self.model.eval()
    
    def unloadModelTokenizer(self):
        before = torch.cuda.memory_allocated() / (1024 * 1024)
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated() / (1024 * 1024)
        self.logger.info(f"{before - after:.2f} MB released from memory")
    
    def embeddingBatch(self, batch, max_length=1024):  
        inputs = self.tokenizer(
            batch, 
            return_tensors='pt',
            truncation=True,
            padding="max_length",
            add_special_tokens=False,
            max_length=max_length)["input_ids"].to(self.device)

        all_hiden_layers = self.model(
                input_ids=inputs, 
                output_hidden_states=True)

        return all_hiden_layers.last_hidden_state
    
    def embeddingProcess(self, batch_size: int = 100):

        sequences = self.dataset[self.column_seq].tolist()

        layer_embeddings = []

        for i in tqdm(range(0, len(sequences), batch_size), desc="[+] Embedding", unit="batch"):
            batch = sequences[i:i + batch_size]

            last_hidden_layer = self.embeddingBatch(batch=batch)

            batch_embedding = torch.mean(last_hidden_layer, dim=1).detach().cpu().numpy()
            layer_embeddings.append(batch_embedding)

        layer_embeddings = np.concatenate(layer_embeddings, axis=0)

        header = [f"p_{i+1}" for i in range(layer_embeddings.shape[1])]

        df_embedding = pd.DataFrame(data=layer_embeddings, columns=header)
        
        for column in self.columns_ignore:
            df_embedding[column] = self.dataset[column].values
        
        return df_embedding