import streamlit as st
import joblib

st.set_page_config(page_title="AI Fake News Classifier", page_icon="📰")
model = joblib.load("fake_news_model.pkl")

st.title("AI-Based Fake News Classification System")
st.write("Enter a news headline or article text for a model prediction.")

news_text = st.text_area("News text", height=180)

if st.button("Classify News"):
    if not news_text.strip():
        st.warning("Please enter some news text first.")
    else:
        prediction = str(model.predict([news_text])[0]).upper()
        if prediction in ["FAKE", "1"]:
            st.error("Model prediction: FAKE")
        elif prediction in ["REAL", "0"]:
            st.success("Model prediction: REAL")
        else:
            st.info("Model prediction: " + prediction)

st.caption(
    "Educational prototype. Predictions may be incorrect; "
    "verify claims with reliable sources."
)