import pandas as pd
import constants
from sklearn.model_selection import train_test_split


class TrainTestPrep:
    def __init__(self, df, test_size = 0.3):
        self.df = df
        self.test_size = test_size
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.y_test = pd.DataFrame()

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df[constants.TEXT],
                                                                                self.df[constants.TARGET],
                                                                                test_size=self.test_size,
                                                                                random_state=1234
                                                                                )
        print('Shape of input data: ', self.df.shape,
              '\nShape of training data: ', self.X_train.shape,
              '\nShape of training data: ', self.X_test.shape)

        print('\nConverting to list..')
        self.X_train = self.X_train.tolist()
        self.X_test = self.X_test.tolist()
        print('\nComplete!..')

        return

