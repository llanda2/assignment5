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
from sklearn.utils import compute_sample_weight

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

# Preprocessing for machine learning
label_encoder = LabelEncoder()
df['VideoCategoryEncoded'] = label_encoder.fit_transform(df['VideoCategory'])

tfidf_vectorizer = TfidfVectorizer()
X_text = df['VideoText']
X_tfidf = tfidf_vectorizer.fit_transform(X_text)

X_combined = pd.concat([pd.DataFrame(X_tfidf.toarray()), df['VideoCategoryEncoded']], axis=1)

# Train-test split on the original data
X_train, X_test, y_train, y_test = train_test_split(X_combined.drop('VideoCategoryEncoded', axis=1), X_combined['VideoCategoryEncoded'], test_size=0.2, random_state=42)

# Create and train a Multinomial Naive Bayes classifier with class weights
classifier = MultinomialNB(class_prior=None, fit_prior=True)
classifier.fit(X_train, y_train, sample_weight=compute_sample_weight(class_weight='balanced', y=y_train))

# Predict the genre of the next video
predictions = classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, predictions)

# Print the accuracy
print(f"Accuracy: {accuracy}")