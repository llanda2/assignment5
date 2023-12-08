import json
import os
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set up API key
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

# Convert video_texts to a 2D array
video_texts_2d = [[text] for text in video_texts]

# Limit the number of videos to match the number of categories
num_videos_to_use = len(video_categories)
video_texts_2d = video_texts_2d[:num_videos_to_use]

# Over-sample minority classes
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(video_texts_2d, [category for category in video_categories.values()])

# Train-test split on the resampled data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Hyperparameter tuning using grid search with StratifiedKFold
param_grid = {
    'pipeline__tfidfvectorizer__max_df': [0.7, 0.8, 0.9],
    'pipeline__tfidfvectorizer__min_df': [1, 2, 3],
    'pipeline__multinomialnb__alpha': [0.1, 0.5, 1.0]
}


pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=stratified_kfold, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best parameters from grid search
print("Best Parameters from Grid Search:", grid_search.best_params_)

# Evaluate the model on the test set
predicted_labels = grid_search.predict(X_test)

# Print accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_test, predicted_labels)
precision = precision_score(y_test, predicted_labels, average='weighted')
recall = recall_score(y_test, predicted_labels, average='weighted')
f1 = f1_score(y_test, predicted_labels, average='weighted')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, predicted_labels)
print("\nConfusion Matrix:")
print(conf_matrix)
