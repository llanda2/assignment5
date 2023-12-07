import json
import os
from collections import Counter
from random import random
import random as rand_module
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# import google_auth_oauthlib.flow
# import googleapiclient.discovery
# import googleapiclient.errors

# Set up  API key
api_key = 'AIzaSyBB__ugeyjrQxGceHF1S6o0PIqBoOfWD4E'

# Fetch video categories using OAuth
scopes = ["https://www.googleapis.com/auth/youtube.readonly"]

# Define stop_words
stop_words = set(stopwords.words('english'))


def fetch_video_categories(api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.videoCategories().list(part="snippet", regionCode="US")
    response = request.execute()
    categories = {item['id']: item['snippet']['title'] for item in response['items']}
    return categories


# Fetch video categories
video_categories = fetch_video_categories(api_key)

# Load YouTube watch history from a JSON file
with open('/Users/laurenlanda/Desktop/Takeout/YouTube and YouTube Music/history/watch-history.json', 'r') as file:
    video_data = json.load(file)

# Extract relevant information
video_texts = [video['title'] + ' ' + video.get('description', '') for video in video_data]


def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)


video_texts = [preprocess_text(text) for text in video_texts]

# Assuming the key for video ID is 'id' in each dictionary within video_data
video_category_mapping = {video.get('id', 'Unknown'): video_categories.get(video.get('id'), 'Unknown') for video in video_data}

# Create a DataFrame with video text and corresponding category
df = pd.DataFrame({'Text': video_texts, 'Category': [video_category_mapping.get(video.get('id'), 'Unknown') for video in video_data]})

# Label Encoding
df['Category_Label'] = pd.Categorical(df['Category']).codes

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Category_Label'], test_size=0.2, random_state=42)

# Use RandomOverSampler to balance classes
sampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X_train.values.reshape(-1, 1), y_train)
X_train_resampled = pd.Series(X_resampled.flatten())

# Text classification model (Naive Bayes as an example)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train_resampled, y_resampled)

# Predict the category for the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
#
# print(f"Accuracy: {accuracy}")
# print("Classification Report:\n", classification_rep)
# print("Confusion Matrix:\n", confusion_mat)
