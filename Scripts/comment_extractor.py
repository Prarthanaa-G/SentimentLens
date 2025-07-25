from googleapiclient.discovery import build
import dotenv
import os
import re
import streamlit as st
from langdetect import detect

dotenv.load_dotenv()

    
def get_english_comments(video_url, max_comments=40):
    api_key = os.getenv("YOUTUBE_API_KEY")
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", video_url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    video_id = match.group(1)

    youtube = build('youtube', 'v3', developerKey=api_key)
    
    
    video_response = youtube.videos().list(
        part='snippet',
        id=video_id
    ).execute()

    if not video_response['items']:
        raise ValueError("Video not found.")

    video_title = video_response['items'][0]['snippet']['title']

    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText"
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
            if len(comments) >= max_comments:
                break

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return video_title,comments

    