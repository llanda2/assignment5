import json
from googleapiclient.discovery import build

# Set up your API key
api_key = 'AIzaSyBB__ugeyjrQxGceHF1S6o0PIqBoOfWD4E'
youtube = build('youtube', 'v3', developerKey=api_key)

# Load your YouTube watch history from a JSON file
with open('/Users/laurenlanda/Desktop/Takeout/YouTube and YouTube Music/history/watch-history.json', 'r') as file:
    watch_history = json.load(file)

# Extract video IDs from the watch history
video_ids = [item['contentDetails']['videoId'] for item in watch_history['items']]

# Get video details to analyze genres
video_details = youtube.videos().list(part='snippet', id=','.join(video_ids)).execute()

# Extract video genres from video details
video_genres = [item['snippet']['categoryId'] for item in video_details['items']]

# Analyze the video genres to predict the next genre (you can customize this part)
most_frequent_genre = max(set(video_genres), key=video_genres.count)

# Print the predicted genre
print(f"The predicted next genre based on your watch history is: {most_frequent_genre}")
