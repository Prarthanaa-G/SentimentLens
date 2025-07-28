# ▶️ SentimentLens : Youtube Comments Sentiment Analyzer 💬

SentimentLens is a powerful tool designed to fetch comments from any YouTube video and perform sentiment analysis on them. It helps users quickly gauge the overall emotional tone of the comments section, classifying feedback into positive, negative, and neutral categories. This tool is perfect for content creators, marketers, and researchers who want to understand audience reception without manually sifting through comments. 

SentimentLens is an end-to-end MLOps project that automates the process of gathering and analyzing sentiment from YouTube comments. It provides a seamless, automated workflow from data collection to a deployed, interactive web application.

## 🎯 The Problem

In the digital age, comment sections are a goldmine of public opinion. However, for any popular video, product review, or educational content, this feedback can amount to thousands or even millions of comments. Manually reading through them is impossible. This leaves:

* **Content Creators** guessing the true reception of their work.
* **Researchers** struggling to gather large-scale qualitative data on public sentiment.
* **Marketers and Brands** unable to quickly gauge reactions to new products, trailers, or campaigns.
* **Educators** missing out on feedback regarding the clarity and impact of their online materials.

SentimentLens solves this by providing a scalable, automated solution to instantly distill vast amounts of text into clear, quantifiable insights.


## ✨ Core Features

* **Fetch Comments**: Extracts all top-level comments from a given YouTube video URL.
* **Sentiment Analysis**: Employs a custom sentiment analysis model, fine-tuned from the distilbert-base-uncased transformer, to accurately classify comments as positive, negative, or neutral.
* **Data Visualization**: Presents the results in a clean, easy-to-understand format with charts showing the sentiment distribution and wordclouds showing most frequently found words in the comments.
* **User-Friendly Interface**: A simple and intuitive web app for anyone to use.

## 🚀 MLOps, CI/CD and Deployment

What makes this project unique is its focus on automation and best practices in MLOps. The entire lifecycle of the application is managed through a robust CI/CD pipeline.

### MLOps Principles
We apply MLOps principles to ensure that our sentiment analysis model and the application are reliable, scalable, and easy to maintain. 

### CI/CD Pipeline
This project is configured with a **GitHub Actions** workflow that automates the following steps upon every push to the `main` branch:

1.  **Code Linting & Testing**: The pipeline first checks the code for style consistency and runs automated tests to prevent regressions.
2.  **Dockerization**: The application (including the backend, model, and frontend) is containerized into a Docker image. This ensures a consistent environment for deployment.
3.  **Push to Hugging Face Spaces**: The newly built Docker container is automatically pushed and deployed to Hugging Face Spaces. This means any update to the code is seamlessly rolled out to the live application without manual intervention.


### 🤗 Hosted on Hugging Face Spaces
The interactive web application is publicly available and hosted on Hugging Face Spaces. This platform is ideal for hosting ML-powered applications, providing the necessary infrastructure and a great user experience.

**[Visit the live application here!](https://huggingface.co/spaces/1Prarthana14/SentimentLens)**



## 🛠️ Tech Stack

* **Backend**: Python
* **Frontend**: Streamlit
* **API**: YouTube Data API v3
* **NLP Library**: Hugging Face Transformers (distilbert-base-uncased fine-tuned)
* **CI/CD**: GitHub Actions
* **Containerization**: Docker
* **Deployment**: Hugging Face Spaces
  

## 📬 Contact

Prarthana G 

Gmail ID: prarthanamurthy29@gmail.com

Linkedin profile: www.linkedin.com/in/prarthana-murthy-581756259


