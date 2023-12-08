# import json
# import os
# import re
# import pandas as pd
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from googleapiclient.discovery import build
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import make_pipeline
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
#
# # Set up API key
# api_key = 'AIzaSyBB__ugeyjrQxGceHF1S6o0PIqBoOfWD4E'
#
# # Fetch video categories using OAuth
# scopes = ["https://www.googleapis.com/auth/youtube.readonly"]
#
# # Define stop_words
# stop_words = set(stopwords.words('english'))
#
#
# # Function to extract video ID from titleUrl
# def extract_video_id(url):
#     if isinstance(url, str):  # Check if 'url' is a string
#         match = re.search(r'v=([a-zA-Z0-9_-]+)', url)
#         return match.group(1) if match else None
#     return None
#
#
# # Function to preprocess text
# def preprocess_text(text):
#     if isinstance(text, str):  # Check if 'text' is a string
#         words = word_tokenize(text)
#         words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
#         return ' '.join(words)
#     return ''
#
#
# # Function to fetch video categories
# def fetch_video_categories(api_key):
#     youtube = build('youtube', 'v3', developerKey=api_key)
#     request = youtube.videoCategories().list(part="snippet", regionCode="US")
#     response = request.execute()
#     categories = {item['id']: item['snippet']['title'] for item in response['items']}
#     return categories
#
#
# # Fetch video categories
# video_categories = fetch_video_categories(api_key)
#
# # Print video categories for debugging
# print("Video Categories:", video_categories)
# # Load JSON data
# with open('/Users/laurenlanda/Desktop/Takeout/YouTube and YouTube Music/history/watch-history.json', 'r') as file:
#     video_data = json.load(file)
#
# # Create a DataFrame from the JSON data
# df = pd.DataFrame(video_data)
#
# # Extract video ID from titleUrl
# df['VideoID'] = df['titleUrl'].apply(extract_video_id)
#
# # Preprocess video texts
# df['VideoText'] = df['title'] + ' ' + df.get('description', '')
# df['VideoText'] = df['VideoText'].apply(preprocess_text)
#
# # Create a new column for video categories using the previously fetched categories
# df['VideoCategory'] = df['VideoID'].replace(video_categories)
#
# # Debug prints for checking the mapping
# print("Mapping of VideoID to VideoCategory:")
# print(df[['VideoID', 'VideoCategory']].head())
# print("Unique Video IDs After Mapping:", df['VideoID'].unique())
#
# # Debug print for checking video_categories
# print("Fetched Video Categories:", video_categories)
# # Debug print to compare VideoID values
# print("VideoID values in DataFrame:", df['VideoID'].unique())
# print("VideoID values in video_categories:", video_categories.keys())

import json
import os
import re
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from googleapiclient.discovery import build
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set up API key
api_key = 'AIzaSyBB__ugeyjrQxGceHF1S6o0PIqBoOfWD4E'

# Fetch video categories using OAuth
scopes = ["https://www.googleapis.com/auth/youtube.readonly"]

# Define stop_words
stop_words = set(stopwords.words('english'))


# Function to extract video ID from titleUrl
def extract_video_id(url):
    if isinstance(url, str):  # Check if 'url' is a string
        match = re.search(r'v=([a-zA-Z0-9_-]+)', url)
        return match.group(1) if match else None
    return None


# Function to preprocess text
def preprocess_text(text):
    if isinstance(text, str):  # Check if 'text' is a string
        words = word_tokenize(text)
        words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
        return ' '.join(words)
    return ''


# Function to fetch video categories
def fetch_video_categories(api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.videoCategories().list(part="snippet", regionCode="US")
    response = request.execute()
    categories = {item['id']: item['snippet']['title'] for item in response['items']}
    return categories


# Fetch video categories
video_categories = fetch_video_categories(api_key)

# Print video categories for debugging
print("Video Categories:", video_categories)

# Load JSON data
with open('/Users/laurenlanda/Desktop/Takeout/YouTube and YouTube Music/history/watch-history.json', 'r') as file:
    video_data = json.load(file)

# Create a DataFrame from the JSON data
df = pd.DataFrame(video_data)

# Extract video ID from titleUrl
df['VideoID'] = df['titleUrl'].apply(extract_video_id)

# Preprocess video texts
df['VideoText'] = df['title'] + ' ' + df.get('description', '')
df['VideoText'] = df['VideoText'].apply(preprocess_text)

# Create a new column for video categories using the previously fetched categories
df['VideoCategory'] = df['VideoID'].replace(video_categories)

# Encode the 'VideoCategory' column into numerical labels
label_encoder = LabelEncoder()
df['Category_Label'] = label_encoder.fit_transform(df['VideoCategory'])

# Train-test split for the classifier
X_train, X_test, y_train, y_test = train_test_split(df['VideoText'], df['VideoCategory'], test_size=0.2, random_state=42)

# Create a pipeline with a TF-IDF vectorizer and a Multinomial Naive Bayes classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))