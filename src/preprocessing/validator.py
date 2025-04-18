import pandas as pd

from constants import LIST_RESIDUES
from logger import get_logger

class ProteinDataValidator(object):

    def __init__(
            self, 
            dataset=None,
            sequence_column=None,
            ignore_columns=None,
            max_length=1024) -> None:
        
        self.dataset = dataset
        self.sequence_column = sequence_column
        self.ignore_columns = ignore_columns
        self.max_length = max_length

        self.status = True
        self.message = ""
                
        self.coded_dataset = pd.DataFrame()
        for column in self.ignore_columns:
            self.coded_dataset[column] = self.dataset[column].values
        
        self.logger = get_logger("ProteinDataValidator")

        self.make_revisions()
        
    def make_revisions(self):

        if self.sequence_column not in self.dataset.columns:
            self.status = False
            self.message = "Non sequence column identified on dataset columns"
        else:
            self.check_canonical_residues()
            self.process_length_sequences()

    def check_canonical_residues(self):
        
        self.logger.info("Checking canonical residues in the dataset")
        canon_sequences = []

        for index in self.dataset.index:
            is_canon=True

            sequence = self.dataset[self.sequence_column][index]

            for residue in sequence:
                if residue not in LIST_RESIDUES:
                    is_canon = False
                    break
            
            canon_sequences.append(is_canon)
        
        self.dataset = self.dataset.copy()
        self.dataset["is_canon"] = canon_sequences
        self.dataset = self.dataset[self.dataset["is_canon"]]
    
    def process_length_sequences(self):
        self.logger.info("Estimating length in protein sequences")
        self.dataset["length_sequence"] = self.dataset[self.sequence_column].str.len()
        
        self.logger.info("Evaluating length in protein sequences")
        self.dataset["is_valid_length"] = (self.dataset["length_sequence"]<=self.max_length).astype(int).values
        self.dataset = self.dataset[self.dataset["is_valid_length"]==1]