import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import joblib
import constants
from src import data_cleaning, text_preprocessing
import definitions
import streamlit as st

# __name__ : from which point execution should begin
clf = joblib.load(os.path.join(definitions.ROOT_DIR, definitions.DATA_DIR, 'model.pkl'))


def predict_sentiment(df):
    # clean data
    clean = data_cleaning.CleanData(df)
    clean.clean_data()

    # text preprocessing
    preprocessed = text_preprocessing.TextPreprocessing(clean.df)
    preprocessed.preprocess()

    bert = SentenceTransformer('all-mpnet-base-v2')
    df = bert.encode(preprocessed.df[constants.TEXT], show_progress_bar=True)

    prediction = clf.predict(df)
    return prediction


def main():
    st.title("Twitter Hate Comment Identifier")
    html_temp = """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">Streamlit Hate Comment Sentiment Analysis ML App </h2>
        </div>
        """

    st.markdown(html_temp, unsafe_allow_html=True)

    tweet = st.text_input("Tweet", "Type Here")
    df_tweet = pd.DataFrame(data=[tweet], columns=[constants.TEXT])
    print(df_tweet)
    result = ""

    if st.button("Predict"):
        result = predict_sentiment(df_tweet)
        if result==0:
            result='Positive'
        else:
            result='Negative'
    st.success('The output is {}'.format(result))

    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")


if __name__ == '__main__':
    main()

# for running:  streamlit run streamlit_app.py

