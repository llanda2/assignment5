import json
from datetime import datetime

from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors

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

# Limit the number of videos to match the number of categories
num_videos_to_use = len(video_categories)
video_texts = video_texts[:num_videos_to_use]

# Train a simple classifier using video categories
X_train, X_test, y_train, y_test = train_test_split(video_texts, [category for category in video_categories.values()],
                                                    test_size=0.2, random_state=42)

classifier = make_pipeline(TfidfVectorizer(), MultinomialNB())
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# # Print the predicted categories
# # print("Predicted Categories for the Next Video:")
# # for predicted_category in y_pred:
# #     print(predicted_category)
# Print all possible genres
print("All Possible Genres:")
for category_id, category_title in video_categories.items():
    print(f"{category_title}: {category_id}")

