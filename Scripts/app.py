import streamlit as st
import pandas as pd
import pickle
import os
os.environ["USE_TF"] = "0"
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

from dotenv import load_dotenv
from helper_functions import log_info, log_error

# Load environment variables
load_dotenv()

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, os.getenv("ARTIFACTS_DIR", "Artifacts"))
# MODEL_PATH = os.path.join(ARTIFACTS_DIR, "logistic_regression_model.pkl")
# PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, "text_pipeline.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")

# Load artifacts
def load_artifact(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading artifact: {filepath}")
        log_error(f"Error loading artifact at {filepath}: {e}")
        return None

# def predict_sentiment(text):
#     pipeline = load_artifact(PIPELINE_PATH)
#     model = load_artifact(MODEL_PATH)
#     label_map = load_artifact(LABEL_ENCODER_PATH)

#     if not pipeline or not model or not label_map:
#         return "Error"

#     transformed = pipeline.transform([text])
#     prediction = model.predict(transformed)[0]

#     reverse_map = {v: k for k, v in label_map.items()}
#     return reverse_map.get(prediction, "Unknown")

def predict_sentiment(text):
    # Load BERT model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained(ARTIFACTS_DIR)
    tokenizer = DistilBertTokenizerFast.from_pretrained(ARTIFACTS_DIR)
    label_encoder = load_artifact(LABEL_ENCODER_PATH)
    if not model or not tokenizer or not label_encoder:
        return "Error"

    # Tokenize input
    inputs = tokenizer([text], truncation=True, padding=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        pred_idx = torch.argmax(outputs.logits, dim=1).item()
    sentiment = label_encoder.inverse_transform([pred_idx])[0]
    return sentiment

# Streamlit UI
st.title("💬 Sentiment Predictor")
user_input = st.text_area("Enter your text for sentiment analysis:", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter text before predicting.")
    else:
        sentiment = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: **{sentiment.capitalize()}**")
        log_info(f"Predicted Sentiment: {sentiment}")
