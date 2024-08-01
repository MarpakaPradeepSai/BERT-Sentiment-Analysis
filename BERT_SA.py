import streamlit as st
from transformers import pipeline

# Load the pre-trained sentiment analysis model
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Define a function to analyze sentiment
def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    sentiment = result[0]['label']
    confidence = result[0]['score']
    return sentiment, confidence

# Streamlit application
st.title('Sentiment Analysis with BERT')

# Text input from the user
text = st.text_area("Enter text for sentiment analysis:", "")

if text:
    # Perform sentiment analysis
    sentiment, confidence = analyze_sentiment(text)
    
    # Display the results
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Confidence: {confidence:.2f}")
