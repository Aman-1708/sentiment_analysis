import pandas as pd
import constants
from sklearn.model_selection import train_test_split


class TrainTestPrep:
    def __init__(self, df, test_size=0.3):
        self.df = df
        self.test_size = test_size

    def train_test(self):
        X_train, X_test, y_train, y_test = train_test_split(self.df[constants.TEXT],
                                                            self.df[constants.TARGET],
                                                            test_size=self.test_size,
                                                            random_state=1234
                                                            )
        print('Shape of input data: ', self.df.shape,
              '\nShape of training data: ', X_train.shape,
              '\nShape of training data: ', X_test.shape)

        print('\nConverting to list..')
        X_train = X_train.tolist()
        X_test = X_test.tolist()
        y_train = y_train.tolist()
        y_test = y_test.tolist()

        print('\nComplete!..')

        return X_train, X_test, y_train, y_test

