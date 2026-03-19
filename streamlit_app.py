import pickle

import streamlit as st

st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬")
st.title("Sentiment Analyzer 🎬")
st.write(
    "Powered by a custom Logistic Regression model trained on 50,000 IMDB reviews."
)


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        saved_data = pickle.load(f)
    return saved_data["vectorizer"], saved_data["model"]


vectorizer, model = load_model()

user_input = st.text_area(
    "Enter a movie review to analyze:",
    height=150,
    placeholder="I thought this movie was...",
)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first!")
    else:
        vectorized_text = vectorizer.transform([user_input])

        prediction = model.predict(vectorized_text)[0]

        if prediction == "positive":
            st.success(f"**Result:** Positive Sentiment! 🍿👍")
        else:
            st.error(f"**Result:** Negative Sentiment! 🍅👎")
