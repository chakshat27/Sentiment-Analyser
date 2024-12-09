import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer

import seaborn as sns
import nltk

# Fix Matplotlib data path issue
import matplotlib
import os
os.environ['MATPLOTLIBDATA'] = matplotlib.get_data_path()
import matplotlib.pyplot as plt

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Streamlit app
st.title("Sentiment Analysis App (User Input)")

# User input
user_input = st.text_input("Enter a sentence:")

if user_input:
    # Perform sentiment analysis
    sentiment_scores = sia.polarity_scores(user_input)

    # Interpret the sentiment scores
    if sentiment_scores['compound'] > 0.35:
        sentiment = "Positive"
    elif sentiment_scores['compound'] < -0.25:
        sentiment = "Negative"
    else:
        sentiment = 'Neutral'

    st.write("Sentiment:", sentiment)
    st.write("Sentiment Scores:", sentiment_scores)



st.markdown(
    """
    <hr>
    <footer style="text-align: center; font-size: small; color: gray;">
        © 2023 Sentiment Analyser App. All rights reserved. | Created By- Chakshat Bali , Savi Chopra
    </footer>
    """,
    unsafe_allow_html=True
)
