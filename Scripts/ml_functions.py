import os
os.environ["USE_TF"] = "0"
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from helper_functions import log_info, log_error
from transformers import DistilBertTokenizerFast, Trainer, TrainingArguments,DistilBertTokenizerFast,DistilBertForSequenceClassification
import torch
import numpy as np
import streamlit as st
from collections import namedtuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'Artifacts')
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")




@st.cache_resource
def load_artifacts():
    """Loads and caches the model, tokenizer, and label encoder."""
    try:
        
        model = DistilBertForSequenceClassification.from_pretrained(ARTIFACTS_DIR)
        log_info("Model loaded and quantized dynamically using PyTorch.")
        tokenizer = DistilBertTokenizerFast.from_pretrained(ARTIFACTS_DIR)
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
            
        log_info("Model, tokenizer, and label encoder loaded successfully.")
        return model, tokenizer, label_encoder
    except Exception as e:
        log_error(f"Error loading artifacts: {e}")
        st.error(f"Fatal error: Could not load ML model artifacts. Please check logs. Error: {e}")
        return None, None, None

  
    
def training_pipeline(X_train, y_train):
    try:
        X_train = X_train[:10000]
        y_train = y_train[:10000]
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=len(set(y_train))
        )

        # Tokenize the data
        train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item
            def __len__(self):
                return len(self.labels)
        train_dataset = CustomDataset(train_encodings, list(y_train))

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=5,
            per_device_train_batch_size=8,
            logging_steps=100,
            save_steps=1000,
            save_total_limit=1,
            logging_dir='./logs',
            disable_tqdm=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset
        )

        trainer.train()
        # Save model and tokenizer
        model.save_pretrained(ARTIFACTS_DIR)
        tokenizer.save_pretrained(ARTIFACTS_DIR)
        log_info(f"BERT model trained and saved at {ARTIFACTS_DIR}")
        return model
    except Exception as e:
        log_error(f"Error during BERT training: {e}")
        raise


def prediction_pipeline(comments,model,tokenizer,label_encoder):
    """
    Performs sentiment analysis on a list of comments using the pre-loaded model.
    """
    if not all([model, tokenizer, label_encoder]):
         return ["Error: Model not loaded"] * len(comments)

    try:
        # Tokenize all comments
        inputs = tokenizer(comments, truncation=True, padding=True, max_length=128, return_tensors='pt')

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            pred_idxs = torch.argmax(outputs.logits, dim=1).numpy()
        
        # Decode predictions
        sentiments = label_encoder.inverse_transform(pred_idxs)
        return sentiments
    except Exception as e:
        log_error(f"Error during prediction: {e}")
        return [f"Error: {e}"] * len(comments)
    
    

def evaluation_metrics(X_val, y_val_encoded, batch_size=32):
    try:      
        model, tokenizer, label_encoder = load_artifacts()

        preds_encoded = []
        for i in range(0, len(X_val), batch_size):
            batch_texts = list(X_val[i:i+batch_size])
            encodings = tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**encodings)
                batch_preds = torch.argmax(outputs.logits, dim=1).numpy()
                preds_encoded.extend(batch_preds)

        preds_encoded = np.array(preds_encoded)
        preds_decoded = label_encoder.inverse_transform(preds_encoded)
        true_labels_decoded = label_encoder.inverse_transform(y_val_encoded)

        conf_matrix = confusion_matrix(true_labels_decoded, preds_decoded, labels=label_encoder.classes_)
        acc_score = accuracy_score(true_labels_decoded, preds_decoded)
        class_report = classification_report(true_labels_decoded, preds_decoded)
        log_info("Evaluation metrics computed successfully.")
        return conf_matrix, acc_score, class_report
    except Exception as e:
        log_error(f"Evaluation failed: {e}")
        raise

