import pandas as pd
import os
from flask import Flask, request
from sentence_transformers import SentenceTransformer
import joblib
import constants
from src import data_cleaning, text_preprocessing
import definitions

# __name__ : from which point execution should begin
app = Flask(__name__)
clf = joblib.load(os.path.join(definitions.ROOT_DIR, definitions.DATA_DIR, 'model.pkl'))


@app.route('/')
def welcome():
    return "Sentiment Analysis to Identify Hate Comments on Twitter"


@app.route('/predict_file', methods=['POST'])
def predict_sentiment():
    df_tweet = pd.read_csv(request.files.get("file"))

    # clean data
    clean = data_cleaning.CleanData(df_tweet[constants.TEXT])
    clean.clean_data()

    # text preprocessing
    preprocessed = text_preprocessing.TextPreprocessing(clean.df)
    preprocessed.preprocess()

    bert = SentenceTransformer('all-mpnet-base-v2')
    df = bert.encode(preprocessed.df, show_progress_bar=True)

    prediction = clf.predict(df)
    return "The prediction is" + str(list(prediction))


if __name__ == '__main__':
    app.run()
