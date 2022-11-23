import pandas as pd
import os
import definitions
import constants


# using data from kaggle link: https://www.kaggle.com/code/abdallahsaadelgendy/twitter-sentiment-analysis-with-lstm/data
# keeping same column names as in the link meta description
class LoadData:
    def __init__(self):
        self.filepath = os.path.join(definitions.ROOT_DIR, definitions.DATA_DIR, definitions.FILE)
        self.col = constants.raw_data_columns
        self.df = pd.DataFrame()

    def load_data(self):
        if '.csv' in definitions.FILE:
            self.df = pd.read_csv(self.filepath,
                                  header=None,
                                  names=self.col)
            print("Data loaded with shape: ", self.df.shape)
            return self.df
        else:
            self.df = pd.read_xlsx(self.filepath,
                                   header=None,
                                   names=self.col)
            print("Data loaded with shape: ", self.df.shape)
            return self.df

