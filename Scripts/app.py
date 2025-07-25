import streamlit as st
import pandas as pd
import pickle
import os
os.environ["USE_TF"] = "0"
import torch
from ml_functions import prediction_pipeline,load_artifacts,evaluation_metrics
from comment_extractor import get_english_comments
from dotenv import load_dotenv
from helper_functions import log_info, log_error
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="YouTube Sentiment Analyzer", layout="centered")
st.title("📺 :red[SentimentLens] ")
st.markdown(
    f"<h2 style='color:black'>YouTube Comments Sentiment Analyzer 💬</h2>",
    unsafe_allow_html=True
)
st.write("---")

load_dotenv()

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, os.getenv("ARTIFACTS_DIR", "Artifacts"))
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")
model, tokenizer, label_encoder = load_artifacts()


# @st.cache_data tells Streamlit to remember the output for a given input URL.
@st.cache_data
def run_full_analysis(video_url):
    """Fetches comments and runs predictions. The results of this function will be cached."""
    log_info(f"Cache miss. Running full analysis for: {video_url}")
    video_title, comments = get_english_comments(video_url)
    if not comments:
        return None, None
    
    predictions = prediction_pipeline(comments, model, tokenizer, label_encoder)
    return video_title,comments, predictions


# Streamlit UI


user_input = st.text_input("Enter a YouTube video URL:", "")
if st.button("Analyze Comments"):
    if user_input.strip() == "" or model is None:
        if user_input.strip() == "":
            st.warning("Please enter a YouTube video URL before analyzing.")
        else:
            st.error("Model artifacts not loaded. Please check logs.")
    else:
        with st.spinner(text=" Fetching Comments and Analyzing sentiment..."):
            try:
                video_title, comments, predictions = run_full_analysis(user_input)

                if not comments:
                    st.warning("No English comments found or comments may be turned off.")
                else:
                    st.success("Analysis complete!")
                    st.write("---")
                    log_info(f"Analyzed {len(comments)} comments for: {user_input}")
                    
                    counts = Counter(predictions)                   
                    st.subheader(f"📊 Sentiment Distribution")
                    st.markdown(f"▶️ <span style='color:gray'>*{video_title}*</span>", unsafe_allow_html=True)
                    st.write("**Counts:**")
                    st.write(f"**- Positive:** {counts.get('positive', 0)}")
                    st.write(f"**- Neutral:** {counts.get('neutral', 0)}")
                    st.write(f"**- Negative:** {counts.get('negative', 0)}")

                    
                    labels = list(counts.keys())
                    values = list(counts.values())
                    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
                    fig.update_traces(marker=dict(line=dict(color='#000000', width=2)), pull=[0.05]*len(labels))
                    st.plotly_chart(fig,use_container_width=True)
                    
                    st.write("---")
                    st.subheader("☁️ Word Clouds of All Comments")
                    text = " ".join(comments)
                    if text:
                        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                        st.image(wordcloud.to_array())
                    
                    # Display individual comments in an expander
                    st.write("---")
                    with st.expander("View Analyzed Comments"):
                        for comment, sentiment in zip(comments, predictions):
                            st.markdown(f"**Sentiment:** `{sentiment}`")
                            st.markdown(f"> {comment}")
                            st.markdown("---")
          
            except Exception as e:
                st.error(f"Failed to analyze video. Reason: {e}")
                log_error(f"Error analyzing video URL {user_input}: {e}")


        




# if __name__ == "__main__":

#     df = pd.read_csv(os.path.join(BASE_DIR, "Data\\raw\\test.csv"))
#     df = df.head(200)
#     X_val = df["text"]

#     # Load label encoder and encode labels
#     label_encoder = pickle.load(open(LABEL_ENCODER_PATH, "rb"))
#     y_val_encoded = label_encoder.transform(df["sentiment"].values)
#     print("Validation data loaded successfully.")

#     conf_matrix, acc_score, class_report = evaluation_metrics(X_val, y_val_encoded)
#     print("Confusion Matrix:\n", conf_matrix)
#     print("Accuracy Score:", acc_score)
#     print("Classification Report:\n", class_report)