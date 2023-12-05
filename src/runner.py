import json
from collections import Counter

from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Set up your API key
api_key = 'AIzaSyBB__ugeyjrQxGceHF1S6o0PIqBoOfWD4E'
youtube = build('youtube', 'v3', developerKey=api_key)

# Load your YouTube watch history from a JSON file
with open('/Users/laurenlanda/Desktop/Takeout/YouTube and YouTube Music/history/watch-history.json', 'r') as file:
    watch_history = json.load(file)

# Extract video IDs from the watch history
video_ids = [item['titleUrl'].split('=')[-1] for item in watch_history]

# Extract video details to analyze
video_data = [{'title': item['title'], 'length': item.get('lengthSeconds', 0), 'time': item['time']} for item in watch_history]

# Extract video genres from video details
video_genres = [item['snippet']['categoryId'] for item in video_details['items']]

# Extract titles for TF-IDF analysis
video_titles = [item['title'] for item in video_data]

# TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(video_titles)

# Calculate cosine similarity between vectors
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Scale video lengths to be between 0 and 1
scaler = MinMaxScaler()
video_lengths = scaler.fit_transform([[item['length']] for item in video_data])

# Scale time periods to be between 0 and 1
video_times = scaler.fit_transform([[item['time']] for item in video_data])

# Combine cosine similarities, video lengths, and time periods into a weighted average
weighted_average = 0.7 * cosine_similarities + 0.2 * video_lengths + 0.1 * video_times

# Calculate the probabilities of each genre based on the weighted average
total_videos = len(video_genres)
genre_probabilities = Counter(video_genres)

# Print the probabilities for the next video genre given a specific genre (customize or optimize)
given_genre = 'gaming'
print(f"Given a genre '{given_genre}', the probability that the next video genre that will be watched is:")
for genre, count in genre_probabilities.items():
    probability = count / total_videos * 100
    print(f"{probability:.2f}% {genre}")