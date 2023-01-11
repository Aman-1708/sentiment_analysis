# Sentiment Analysis with Streamlit App and Docker Deployment

**Objective**<br>
The objective of this task is to detect hate speech in tweets. For the sake of simplicity, we say a tweet contains hate speech if it has a racist or sexist sentiment associated with it. So, the task is to classify racist or sexist tweets from other tweets. (Class 1: Hate speech, 0 otherwise)

Streamlit and docker image are developed for model deployment.

Data source: https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech?select=train.csv

**Instructions**<br>
* Run main.py for model training
* Create docker image using the command in the below section
* Execute!

### Reference Code Sippets
**Freeze requirements**<br>
command: pip3 freeze > requirements.txt

**Command to run Streamlit app** <br>
streamlit run streamlit_app.py

**Command to Build Docker image**<br>
docker build -t sentiment_api .

**Command to Run Docker Image** <br>
docker run -p 8501:8501 sentiment_app