# importing libraries
import constants
import joblib
import definitions
import os
from src import load_data, data_cleaning, text_preprocessing, embedding, train_model, train_test_data_prep

import pandas as pd
pd.set_option('display.max_columns', None)

# Model Training------------------------------------------------------------------------------------
# load data
data = load_data.LoadData(os.path.join(definitions.ROOT_DIR, definitions.DATA_DIR, definitions.TRAIN_FILE))
data.load_data()

# clean data
clean = data_cleaning.CleanData(data.df)
clean.clean_data()

# text preprocessing
preprocessed = text_preprocessing.TextPreprocessing(clean.df)
preprocessed.preprocess()

# train-test split
train_test_data = train_test_data_prep.TrainTestPrep(preprocessed.df)
X_train, X_test, y_train, y_test = train_test_data.train_test()

# bert embeddings
encode_data = embedding.Embedding(X_train, X_test)
encode_data.encode()

# train model
model = train_model.TrainModel()
model.train_model(X_train=encode_data.X_train,
                  y_train=y_train,
                  X_test=encode_data.X_test,
                  y_test=y_test)

print(model.clf)

joblib.dump(model.clf, os.path.join(definitions.ROOT_DIR, definitions.DATA_DIR, 'model.pkl'))