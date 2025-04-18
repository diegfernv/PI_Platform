import math 
import numpy as np
import pandas as pd
from scipy.fft import fft
from logger import get_logger

class FFTEncoder(object):

    def __init__(
            self, 
            dataset=None, 
            sequence_column=None, 
            ignore_columns=[]) -> None:
        
        self.dataset = dataset
        self.sequence_column = sequence_column
        self.ignore_columns = ignore_columns
        self.max_length = len(self.dataset.columns) - len(self.ignore_columns)

        self.logger = get_logger("FFTEncoder")

        self.init_process()
        self.df_fft = None

    def __processing_data_to_fft(self):

        self.logger.info("Removing columns data")
        
        if len(self.ignore_columns) >0:
            self.data_ignored = self.dataset[self.ignore_columns]
            self.dataset = self.dataset.drop(columns=self.ignore_columns)
    
    def __get_near_pow(self):

        self.logger.info("Getting near pow 2 value")

        list_data = [math.pow(2, i) for i in range(1, 20)]
        stop_value = list_data[0]

        for value in list_data:
            if value >= self.max_length:
                stop_value = value
                break

        self.stop_value = int(stop_value)

    def __complete_zero_padding(self):

        self.logger.info("Applying zero padding")
        list_df = [self.dataset]
        for i in range(self.max_length, self.stop_value):
            column = [0 for k in range(len(self.dataset))]
            key_name = "p_{}".format(i)
            df_tmp = pd.DataFrame()
            df_tmp[key_name] = column
            list_df.append(df_tmp)

        self.dataset = pd.concat(list_df, axis=1)
    

    def init_process(self):
        self.__processing_data_to_fft()
        self.__get_near_pow()
        self.__complete_zero_padding()

    def __create_row(self, index):
        row =  self.dataset.iloc[index].tolist()
        return row
    
    def __apply_FFT(self, index):

        row = self.__create_row(index)
        T = 1.0 / float(self.stop_value)
        yf = fft(row)

        xf = np.linspace(0.0, 1.0 / (2.0 * T), self.stop_value // 2)
        yf = np.abs(yf[0:self.stop_value // 2])
        return [value for value in yf]


    def encoding_dataset(self):

        matrix_response = []
        for index in self.dataset.index:
            row_fft = self.__apply_FFT(index)
            matrix_response.append(row_fft)

        self.logger.info("Creating dataset")
        header = ['p_{}'.format(i) for i in range(len(matrix_response[0]))]
        self.logger.info("Exporting dataset")
        self.df_fft = pd.DataFrame(matrix_response, columns=header)
        
        if len(self.ignore_columns)>0:

            self.df_fft = pd.concat([self.df_fft, self.data_ignored], axis=1)
