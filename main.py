# importing libraries
from src import load_data, data_cleaning, text_preprocessing, train_test_data_prep, embedding, train_model

# load data
data = load_data.LoadData()
data.load_data()

# clean data
clean = data_cleaning.CleanData(data.df)
clean.clean_data()

# text preprocessing
preprocessed = text_preprocessing.TextPreprocessing(clean.df)
preprocessed.preprocess()

# train test split
train_test_data = train_test_data_prep.TrainTestPrep(preprocessed.df)
train_test_data.train_test_split()

# bert embeddings
encode_data = embedding.Embedding(train_test_data.X_train, train_test_data.X_test)
encode_data.encode()

# train model
model = train_model.TrainModel()
model.train_model(X_train=encode_data.X_train,
                  y_train=train_test_data.y_train,
                  X_test=encode_data.X_test,
                  y_test=train_test_data.y_test)

