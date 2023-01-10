FROM python:3.9.13

COPY . /app

EXPOSE 8501

WORKDIR /app

RUN pip3 install -r requirements.txt

CMD streamlit run streamlit_app.py

