FROM python:3.8-slim-buster
WORKDIR /Sentiment_Analysis
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install -r requirements.txt
COPY sentiment_analysis_lstm.py sentiment_analysis_lstm.py
COPY sentiment_LSTM_model.pth sentiment_LSTM_model.pth
COPY . /Sentiment_Analysis
CMD [ "python3", "sentiment_analysis_lstm.py" ]