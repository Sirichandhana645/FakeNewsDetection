import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import re

nltk.download('stopwords')

# Load the model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Preprocess function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection")
st.markdown("Enter a news article or headline below to check if it's *Fake* or *Real*.")

user_input = st.text_area("🔍 Paste your news text here", height=200)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_text = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_input)[0]

        if prediction == 1:
            st.success("✅ This looks like *Real News*.")
        else:
            st.error("🚨 This appears to be *Fake News*.")