from typing import List
import torch
import pandas as pd
from transformers import BertModel, BertTokenizer

from .EmbeddingsBase import EmbeddingsBase

class BertBased(EmbeddingsBase):

    def __init__(
            self, 
            name_device: str = "cuda", 
            name_model: str = "Rostlab/prot_bert", 
            name_tokenizer: str = "Rostlab/prot_bert",
            dataset: pd.DataFrame = None,
            column_seq: str = None,
            columns_ignore: List[str] = []):
        
        super().__init__(name_device, name_model, name_tokenizer, dataset, column_seq, columns_ignore)
    
    def loadModelTokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            self.name_model, 
            do_lower_case=False)
        
        self.model = BertModel.from_pretrained(self.name_model).to(self.device)

        self.model.eval()

    def unloadModelTokenizer(self):
        before = torch.cuda.memory_allocated() / (1024 * 1024)
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated() / (1024 * 1024)
        self.logger.info(f"{before - after:.2f} MB released from memory")