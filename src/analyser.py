import json
import os
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tabulate import tabulate

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

# Limit the number of videos to match the number of categories
num_videos_to_use = len(video_categories)
video_texts = video_texts[:num_videos_to_use]

# Train a simple classifier using video categories
X_train, X_test, y_train, y_test = train_test_split(video_texts, [category for category in video_categories.values()],
                                                    test_size=0.2, random_state=42)

classifier = make_pipeline(TfidfVectorizer(), MultinomialNB())
classifier.fit(X_train, y_train)

# Define the output directory and file path
output_directory = 'src/out'
output_file_path = os.path.join(output_directory, 'genreResults.txt')

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Predict the next video category for each video in the dataset
with open(output_file_path, 'w') as output_file:
    # Predict the next video category for each video in the dataset
    predictions = []

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

            # Store predictions in a list
            predictions.append({
                'current_category': current_category,
                'predicted_category': predicted_category,
                'probability_percentage': probability_percentage,
                'views': video_id  # Replace this with the actual number of views
            })

    # Sort predictions based on probability in descending order
    sorted_predictions = sorted(predictions, key=lambda x: x['probability_percentage'], reverse=True)

    # Write the sorted predictions to the output file
    with open(output_file_path, 'w') as output_file:
        for prediction in sorted_predictions:
            output_line = (f"Given that the current genre of the video is '{prediction['current_category']}', "
                           f"the likelihood that the next video is '{prediction['predicted_category']}' is {prediction['probability_percentage']:.2f}%\n")
            output_file.write(output_line)

        # Print chart with predicted and previous categories
        chart_data = [(prediction['current_category'], prediction['predicted_category']) for prediction in
                      sorted_predictions]

        output_file.write("\nChart:\n")
        chart_headers = ["Previous", "Predicted"]
        chart_str = tabulate(chart_data, headers=chart_headers, tablefmt="pretty")
        output_file.write(chart_str)

        # Find the most and least watched categories
        most_watched_category = max(sorted_predictions, key=lambda x: x['views'])['predicted_category']
        least_watched_category = min(sorted_predictions, key=lambda x: x['views'])['predicted_category']

        output_file.write(f"\nMost Watched Category: {most_watched_category}\n")
        output_file.write(f"Least Watched Category: {least_watched_category}\n")
