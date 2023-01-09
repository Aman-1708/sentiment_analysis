import pandas as pd
import os
import definitions
import constants


# using data from kaggle:
# https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech?select=train.csv
class LoadData:
    def __init__(self, path):
        self.filepath = path
        self.col = constants.raw_data_columns
        self.df = pd.DataFrame()

    def load_data(self):
        if '.csv' in self.filepath:
            self.df = pd.read_csv(self.filepath)
            print("Data loaded with shape: ", self.df.shape)
            print("Snapshot of Data: \n", self.df.head())
            return self.df
        else:
            self.df = pd.read_xlsx(self.filepath)
            print("Data loaded with shape: ", self.df.shape)
            print("Snapshot of Data: \n", self.df.head())
            return self.df

