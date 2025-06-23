import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# App title
st.set_page_config(page_title="📰 Fake News Detector", layout="centered")
st.title("🧠 Fake News Detection App")
st.markdown("Enter a news article below and find out whether it's **Fake** or **Real**.")

# Input form
news_text = st.text_area("✍️ Paste the news article text here", height=250)

# Predict button
if st.button("🔍 Predict"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        # Transform input and predict
        input_vector = vectorizer.transform([news_text])
        prediction = model.predict(input_vector)[0]

        # Display result
        if prediction == 1:
            st.success("✅ This news article is **Real**.")
        else:
            st.error("❌ This news article is **Fake**.")
