# importing libraries
from src import load_data, data_cleaning, text_preprocessing

# load data
df = load_data.LoadData().load_data()

# clean data
df = data_cleaning.CleanData(df).clean_data()

# text preprocessing
df = text_preprocessing.TextPreprocessing(df).preprocess()

print(df.head())
