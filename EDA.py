# importing libraries
import pandas as pd
import constants
from src import load_data
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

# load data
df = load_data.LoadData().load_data()
print(df.head())

# Checking duplicated rows
print("Number of duplicated rows", df.duplicated().sum())
print("Shape of data before dropping duplicates rows", df.shape)
print("Dropping duplicated rows...")
df.drop_duplicates(inplace=True)
print("Shape of data after dropping duplicates rows", df.shape)

# Checking unique entries to determine level of data
# Multiple tweet id found
print("\nUnique Entries :\n", df.nunique())

# Checking Missing Data
print("\n# of Missing columns: \n", df.isna().sum())
# Removing rows with missing columns
print("Shape of data before dropping missing rows", df.shape)
df.dropna(inplace=True)
print("Shape of data after dropping missing rows", df.shape)

# reset index
df.sort_values(by=['TweetID']).reset_index(inplace=True)

# Checking Sentiment distribution
print("\nTarget Distribution :\n", df[constants.TARGET].value_counts(normalize=True, dropna=False))

# Checking distribution of length of tweet
print(df['Tweet content'].apply(lambda x: len(x)).min())
print(df['Tweet content'].apply(lambda x: len(x)).max())

# data exists with only spaces in tweet content
print(df[df[constants.TEXT].str.isspace()==True])



