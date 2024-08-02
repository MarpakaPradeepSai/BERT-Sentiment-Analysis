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

# Inject custom CSS
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #28a745; /* Green color */
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 5px;
    }

    .stButton > button:hover {
        background-color: #218838; /* Darker green for hover effect */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.title("Sentiment Analysis App")

user_input = st.text_area("Enter text here:")

# Add a button to analyze sentiment
if st.button('Analyze Sentiment'):
    if user_input:
        sentiment, confidence = analyze_sentiment(user_input)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {confidence:.2f}")
    else:
        st.write("Please enter some text.")
