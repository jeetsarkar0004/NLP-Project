from pathlib import Path
import re
import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "logistic_regression_model_final.joblib"


@st.cache_resource
def load_model_and_vectorizer():
    stop_words = set(stopwords.words("english"))

    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        words = nltk.word_tokenize(text)
        cleaned_words = [word for word in words if word not in stop_words and word.isalpha()]
        return " ".join(cleaned_words)

    # Load the saved model and vectorizer
    model_data = joblib.load(MODEL_PATH)
    
    if isinstance(model_data, dict):
        model = model_data['model']
        vectorizer = model_data['vectorizer']
    else:
        # Fallback for old model format
        model = model_data
        vectorizer = TfidfVectorizer(max_features=5000)
    
    return model, vectorizer, stop_words, preprocess_text


model, vectorizer, stop_words, preprocess_text = load_model_and_vectorizer()


def predict_sentiment(review_text: str):
    cleaned_text = preprocess_text(review_text)

    features = vectorizer.transform([cleaned_text])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    probability_map = dict(zip(model.classes_, probabilities))

    # Normalize prediction to lowercase
    label = str(prediction).strip().lower()
    
    # Map to standard labels
    if label not in {"positive", "negative"}:
        if "pos" in label:
            label = "positive"
        elif "neg" in label:
            label = "negative"

    confidence = max(probability_map.values())
    return label, confidence, probability_map


st.set_page_config(page_title="Review Sentiment App", page_icon="📝", layout="centered")
st.title("Review Sentiment Predictor")
st.write("Enter a review and the app will predict whether it is positive or negative.")

review_text = st.text_area(
    "Review text",
    value="I absolutely loved this product and would buy it again.",
    height=140,
)

if st.button("Predict Sentiment"):
    if not review_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        predicted_label, confidence, probability_map = predict_sentiment(review_text)
        
        # Display prediction with color coding
        if predicted_label.lower() == "positive":
            st.success(f"✅ Predicted sentiment: **POSITIVE**")
        else:
            st.error(f"❌ Predicted sentiment: **NEGATIVE**")
        
        st.write(f"Confidence: {confidence:.2%}")

        st.subheader("Probability breakdown")
        # Only show Positive and Negative probabilities
        for label, prob in probability_map.items():
            label_str = str(label).strip()
            if label_str.lower() not in ["neutral"]:
                emoji = "✅" if label_str.lower() == "positive" else "❌"
                st.write(f"{emoji} {label_str.title()}: {prob:.2%}")
