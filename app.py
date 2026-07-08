import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load("logistic_regression.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬"
)

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review below and click Predict.")

# Text input
review = st.text_area("Movie Review")

if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        review_vector = vectorizer.transform([review])
        prediction = model.predict(review_vector)[0]
        probability = model.predict_proba(review_vector)[0]

        if prediction == 1:
            st.success("😊 Positive Review")
            st.write(f"Confidence: {probability[1]*100:.2f}%")
        else:
            st.error("😞 Negative Review")
            st.write(f"Confidence: {probability[0]*100:.2f}%")