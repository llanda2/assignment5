import json
import os
from collections import Counter
from random import random
import random as rand_module
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

# Limit the number of videos to match the number of categories
num_videos_to_use = len(video_categories)
video_texts = video_texts[:num_videos_to_use]

# Train a simple classifier using video categories
X_train, X_test, y_train, y_test = train_test_split(video_texts, [category for category in video_categories.values()],
                                                    test_size=0.2, random_state=42)

classifier = make_pipeline(TfidfVectorizer(), MultinomialNB())
classifier.fit(X_train, y_train)

# Use the trained classifier to make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display additional metrics
classification_rep = classification_report(y_test, y_pred)

# Display confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)

# Define the output directory and file path
output_directory = 'src/out'
output_file_path = os.path.join(output_directory, 'genreResults.txt')

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)
# Validate Samples

# Predict the next video category for each video in the dataset
with open(output_file_path, 'w') as output_file:
    for video_id, current_category in video_categories.items():
        # Convert video_id to integer
        video_id = int(video_id)

        if video_id < len(video_texts):
            current_title = video_texts[video_id]

            # Preprocess the current title
            current_title = preprocess_text(current_title)

            # Make prediction on the current title
            predicted_category_prob = classifier.predict_proba([current_title])[0]
            predicted_category = classifier.classes_[predicted_category_prob.argmax()]
            probability_percentage = predicted_category_prob.max() * 100

            # Write the output to the file
            output_line = (f"Given the genre '{current_category}' of the previous video, "
                           f"the probability of the genre '{predicted_category}' of the next video being watched is {probability_percentage:.2f}%\n")
            output_file.write(output_line)

    # # Write accuracy, classification report, and confusion matrix to the file
    # output_file.write(f"\nAccuracy: {accuracy:.2%}\n\n")
    # output_file.write("Classification Report:\n")
    # output_file.write(str(classification_rep))
    # output_file.write("\n\nConfusion Matrix:\n")
    # output_file.write(str(confusion_mat))