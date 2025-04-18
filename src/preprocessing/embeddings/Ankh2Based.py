from typing import List
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

from .EmbeddingsBase import EmbeddingsBase

class Ankh2Based(EmbeddingsBase):

    def __init__(
            self, 
            name_device: str = "cuda", 
            name_model: str = "ElnaggarLab/ankh2-ext1", 
            name_tokenizer: str= "ElnaggarLab/ankh2-ext1",
            dataset: pd.DataFrame = None,
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


    def embeddingBatch(self, batch, max_length: int = 1024):
        
        inputs = self.tokenizer(
            batch, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=max_length,
            add_special_tokens=False
        ).to(self.device)

        # Forward pass using encoder-only mode
        with torch.no_grad():
            encoder_outputs = self.model.encoder(
                input_ids=inputs["input_ids"],
                output_hidden_states=False
            )
        
        del inputs

        return encoder_outputs.last_hidden_state