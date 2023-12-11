import json
import os

import numpy as np
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from tabulate import tabulate

# Set up API key
api_key = 'AIzaSyBB__ugeyjrQxGceHF1S6o0PIqBoOfWD4E'  # Replace with your actual YouTube API key

# Fetch video categories using YouTube API
# Followed youtube video for setting this up (referenced in README.md #4)
def fetch_video_category(api_key, video_url):
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Split video_url to extract video ID
    video_url_parts = video_url.split("v=")

    # Check if there are at least two parts after the split
    if len(video_url_parts) > 1:
        video_id = video_url_parts[1]

        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()

        # Check if there are items in the response
        if 'items' in response and response['items']:
            # Extract category from the response
            category_id = response['items'][0]['snippet']['categoryId']
            return category_id
    # Handle the case where the split didn't provide a valid video ID
    return None


# Load YouTube watch history from a JSON file
with open('/Users/laurenlanda/Desktop/Takeout/YouTube and YouTube Music/history/watch-history.json', 'r') as file:
    watch_history = json.load(file)
# Use only the first 1000 entries for testing
watch_history_subset = watch_history[:1000]

# Extract relevant information from the subset
video_texts = [video['title'] + ' ' + video.get('description', '') for video in watch_history_subset]
video_urls = [video.get('titleUrl', '') for video in watch_history_subset if 'titleUrl' in video]
video_categories = [fetch_video_category(api_key, url) for url in video_urls]

# Perform random sampling
random_indices = np.random.permutation(len(video_texts))
video_texts = [video_texts[i] for i in random_indices]
video_categories = [video_categories[i] for i in random_indices]

# Train a simple classifier using video categories
X_train, X_test, y_train, y_test = train_test_split(video_texts, video_categories, test_size=0.2, random_state=42)

# Filter out instances with None in the target variable
X_train = [text for text, category in zip(X_train, y_train) if category is not None]
y_train = [category for category in y_train if category is not None]

# Create a pipeline with TF-IDF vectorizer and Naive Bayes classifier
classifier = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the classifier
classifier.fit(X_train, y_train)

# Update the category_mapping dictionary based on the actual category IDs
category_mapping = {
    '1': 'Film & Animation',
    '2': 'Autos & Vehicles',
    '10': 'Music',
    '15': 'Pets & Animals',
    '17': 'Sports',
    '18': 'Short Movies',
    '19': 'Travel & Events',
    '20': 'Gaming',
    '21': 'Videoblogging',
    '22': 'People & Blogs',
    '23': 'Comedy',
    '24': 'Entertainment',
    '25': 'News & Politics',
    '26': 'Howto & Style',
    '27': 'Education',
    '28': 'Science & Technology',
    '29': 'Nonprofits & Activism',
    '30': 'Movies',
    '31': 'Anime/Animation',
    '32': 'Action/Adventure',
    '33': 'Classics',
    '34': 'Comedy',
    '35': 'Documentary',
    '36': 'Drama',
    '37': 'Family',
    '38': 'Foreign',
    '39': 'Horror',
    '40': 'Sci-Fi/Fantasy',
    '41': 'Thriller',
    '42': 'Shorts',
    '43': 'Shows',
    '44': 'Trailers'
}

# Predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the accuracy
accuracy = sum(y_pred == y_test) / len(y_test)
print(f"Accuracy on Test Data: {accuracy:.2%}")

# Define the output directory and file path
output_directory = 'src/out'
output_file_path = os.path.join(output_directory, 'genreResults.txt')

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Write the unique predictions to the output file
with open(output_file_path, 'w') as output_file:
    # Predict the next video category for each video in the dataset
    predictions = []

    for video_id, current_category_id in enumerate(y_test):
        # Get the current title
        current_title = X_test[video_id]

        # Make prediction on the current title with probabilities
        predicted_category_prob = classifier.predict_proba([current_title])[0]
        predicted_category = classifier.classes_[predicted_category_prob.argmax()]
        probability_percentage = predicted_category_prob.max() * 100

        # Map category IDs to category names using the globally defined category_mapping
        current_category_name = category_mapping.get(current_category_id, 'Unknown')
        predicted_category_name = category_mapping.get(predicted_category, 'Unknown')

        # Store predictions in a list
        predictions.append({
            'current_category': current_category_name,
            'predicted_category': predicted_category_name,
            'probability_percentage': probability_percentage,
            'views': video_id  # Replace this with the actual number of views
        })

    # Remove duplicate predictions based on 'current_category' and 'predicted_category'
    unique_predictions = []
    seen_combinations = set()

    for prediction in predictions:
        combination = (prediction['current_category'], prediction['predicted_category'])
        if combination not in seen_combinations:
            seen_combinations.add(combination)
            unique_predictions.append(prediction)

    # Sort predictions based on probability in descending order
    sorted_predictions = sorted(unique_predictions, key=lambda x: x['probability_percentage'], reverse=True)

    # Write the sorted predictions to the output file
    for prediction in sorted_predictions:
        output_line = (f"Given that the current genre of the video is '{prediction['current_category']}', "
                       f"the likelihood that the next video is '{prediction['predicted_category']}' is {prediction['probability_percentage']:.2f}%\n")
        output_file.write(output_line)

    # Print chart with actual and predicted categories
    chart_data = [
        (category_mapping.get(current_category_id, 'Unknown'), category_mapping.get(predicted_category, 'Unknown')) for
        current_category_id, predicted_category in zip(y_test, classifier.predict(X_test))]

    # Remove duplicate entries in the chart
    seen_chart_combinations = set()
    unique_chart_data = []

    for entry in chart_data:
        if entry not in seen_chart_combinations:
            seen_chart_combinations.add(entry)
            unique_chart_data.append(entry)

    output_file.write("\nChart:\n")
    chart_headers = ["Actual", "Predicted"]
    chart_str = tabulate(unique_chart_data, headers=chart_headers, tablefmt="pretty")
    output_file.write(chart_str)

    # Find the category watched the most and least
    category_counts = {}
    for entry in unique_chart_data:
        for category in entry:
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1

    most_watched_category = max(category_counts, key=category_counts.get)
    least_watched_category = min(category_counts, key=category_counts.get)

    output_file.write(f"\nMost Watched Category: {most_watched_category}\n")
    output_file.write(f"Least Watched Category: {least_watched_category}\n")

print(f"Results written to: {output_file_path}")
