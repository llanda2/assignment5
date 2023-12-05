import json
from collections import Counter
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

__author__ = "Lauren Landa"
__credits__ = ["Lauren Landa"]
__license__ = "MIT"

# Set up your API key
api_key = 'AIzaSyBB__ugeyjrQxGceHF1S6o0PIqBoOfWD4E'
youtube = build('youtube', 'v3', developerKey=api_key)

# Load your YouTube watch history from a JSON file
with open('/Users/laurenlanda/Desktop/Takeout/YouTube and YouTube Music/history/watch-history.json', 'r') as file:
    watch_history = json.load(file)

video_ids = [item.get('contentDetails', {}).get('videoId', '') for item in watch_history]

# Get video details to analyze genres
video_details = youtube.videos().list(part='snippet', id=','.join(video_ids)).execute()

# Extract video genres from video details
video_genres = [item['snippet']['categoryId'] for item in video_details['items']]

# Extract titles for TF-IDF analysis
video_titles = [item['snippet']['title'] for item in video_details['items']]

# TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(video_titles)

# Calculate cosine similarity between vectors
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Scale video lengths to be between 0 and 1
scaler = MinMaxScaler()
video_lengths = scaler.fit_transform([[item['contentDetails']['duration']] for item in video_details['items']])

# Scale time periods to be between 0 and 1
video_times = scaler.fit_transform([[item['snippet']['publishedAt']] for item in video_details['items']])

# Combine cosine similarities, video lengths, and time periods into a weighted average
weighted_average = 0.7 * cosine_similarities + 0.2 * video_lengths + 0.1 * video_times

# Calculate the probabilities of each genre based on the weighted average
total_videos = len(video_genres)
genre_probabilities = Counter(video_genres)

# Print the probabilities for the next video genre given a specific genre (you can customize this part)
given_genre = 'gaming'
print(f"Given a genre '{given_genre}', the probability that the next video genre that will be watched is:")
for genre, count in genre_probabilities.items():
    probability = count / total_videos * 100
    print(f"{probability:.2f}% {genre}")
