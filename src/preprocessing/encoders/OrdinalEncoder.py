import pandas as pd

from preprocessing.validator import ProteinDataValidator
from constants import POSITION_RESIDUES
from logger import get_logger


class OrdinalEncoder(ProteinDataValidator):

    def __init__(
            self, 
            dataset=None, 
            sequence_column=None, 
            ignore_columns=[],
            max_length=1024) -> None:
        
        super().__init__(
            dataset=dataset, 
            sequence_column=sequence_column, 
            ignore_columns=ignore_columns,
            max_length=max_length)
        
        self.logger = get_logger("OrdinalEncoder")
    
    def __zero_padding(self, current_length):

        zero_padding_vector = [0 for i in range(current_length, self.max_length+1)]
        return zero_padding_vector
    
    def __coding_sequence(self, sequence):

        coded_vector = []

        for residue in sequence:
            coded_vector.append(POSITION_RESIDUES[residue])

        if len(sequence) < self.max_length:
            coded_vector += self.__zero_padding(len(coded_vector))
        
        return coded_vector
    
    def run_process(self):

        if self.status:
            self.logger.info("Starting encoding")
            matrix_coded = []

            for index in self.dataset.index:
                sequence = self.dataset[self.sequence_column][index]

                matrix_coded.append(
                    self.__coding_sequence(sequence)
                )
            
            header = [f"p_{i}" for i in range(len(matrix_coded[0]))]
            self.coded_dataset = pd.DataFrame(data=matrix_coded, columns=header)

            for column in self.ignore_columns:
                self.coded_dataset[column] = self.dataset[column].values
        else:
            self.logger.error("Coded was no possible, please check the configuration")