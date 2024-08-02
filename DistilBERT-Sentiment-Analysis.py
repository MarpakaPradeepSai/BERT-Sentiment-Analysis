import streamlit as st
from transformers import pipeline

# Load the pre-trained sentiment analysis model
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Define a function to analyze sentiment
def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    sentiment = result[0]['label']
    confidence = result[0]['score']
    return sentiment , confidence

# Inject custom CSS for button styling
st.markdown(
    """
    <style>
    .custom-button {
        background-color: #28a745; /* Green color */
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }

    .custom-button:hover {
        background-color: #218838; /* Darker green for hover effect */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.title("Sentiment Analysis App")

user_input = st.text_area("Enter text here:")

# Add a custom button using HTML and JavaScript
if st.markdown('<button class="custom-button" onclick="analyzeSentiment()">Analyze Sentiment</button>', unsafe_allow_html=True):
    if user_input:
        sentiment, confidence = analyze_sentiment(user_input)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {confidence:.2f}")
    else:
        st.write("Please enter some text.")
