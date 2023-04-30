# Import Libraries

import numpy as np
import pandas as pd
import re

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
#import torch.optim as optim
#from torchtext.vocab import Vocab
#from torch.utils.data import Dataset, DataLoader
#from torchtext.data.utils import get_tokenizer
#from torchtext.vocab import build_vocab_from_iterator

from transformers import BertTokenizer
from transformers import AutoTokenizer

text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Create LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, dropout=0.3):
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, input_ids, attention_mask):
        embedded = self.dropout(self.embedding(input_ids))
        packed_output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        output = self.fc(hidden)

        return output
    
    # Hyperparameters
HIDDEN_DIM = 128
EMBED_DIM = 128
DROPOUT = 0.3
N_CLASSES = 4


# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMClassifier(len(text_tokenizer), EMBED_DIM, HIDDEN_DIM, N_CLASSES, DROPOUT).to(device)

model_path = "sentiment_LSTM_model.pth"
model.load_state_dict(torch.load(model_path))
model.eval()


def preprocess_and_tokenize(tweet, tokenizer, max_len):
    tweet = preprocess_text(tweet)  # Assuming you have the preprocess_text function from the previous notebook
    encoded_input = tokenizer.encode_plus(
        tweet,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return encoded_input

def predict_sentiment(tweet, model, tokenizer, device):
    max_len = 128  # Set the same max length used during training
    encoded_input = preprocess_and_tokenize(tweet, tokenizer, max_len)
    
    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    sentiment = np.argmax(outputs.cpu().numpy(), axis=1)
    return sentiment


# Preprocess the text data
def preprocess_text(text):
    text = str(text).lower()  # Convert the text to a string
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\W+', ' ', text)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(tokens)


def sentiment_to_text(sentiment_value):
    sentiment_map = {
        0: 'Positive',
        1: 'Negative',
        2: 'Neutral',
        3: 'Irrelevant'
    }
    return sentiment_map[sentiment_value]

def sentiment_predict(tweet):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    sentiment_value = predict_sentiment(tweet, model, tokenizer, device)
    sentiment_value = sentiment_value.item()  # Convert the numpy array to a single integer value
    sentiment_text = sentiment_to_text(sentiment_value)
    return sentiment_text

import gradio as gr

# Create the Gradio interface
interface2 = gr.Interface(fn=sentiment_predict,  inputs='text', outputs='text', title="Sentiment Analysis")
interface2.launch(inline=True, share = True)