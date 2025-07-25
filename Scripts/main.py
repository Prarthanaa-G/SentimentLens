import os
import pandas as pd
from dotenv import load_dotenv
from data_processing import create_text_pipeline, save_pipeline, encode_response_variable
from ml_functions import training_pipeline,evaluation_metrics
from helper_functions import log_info, log_error
import pickle
from sklearn.model_selection import train_test_split

# Load environment variables
load_dotenv()

# Define base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, os.getenv('DATA_DIR', 'Data'))
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'train.csv')
ARTIFACTS_DIR = os.path.join(BASE_DIR, os.getenv('ARTIFACTS_DIR', 'Artifacts'))
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "logistic_regression_model.pkl")

# Ensure artifacts directory exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def main():
    log_info("🚀 Starting Sentiment Analysis Training Pipeline")

    try:
        train_df = pd.read_csv(RAW_DATA_PATH)
        log_info(f"✅ Loaded training data: {train_df.shape}")
    except Exception as e:
        log_error(f"❌ Failed to load training data: {e}")
        return

    X_raw = train_df["text"].fillna("")
    y = train_df["sentiment"].fillna("")

    y_encoded = encode_response_variable(y)

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_raw, y_encoded, test_size=0.2, random_state=42)


    model = training_pipeline(X_train_raw, y_train)


if __name__ == "__main__":
    main()
