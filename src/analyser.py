# import json
# import os
# from googleapiclient.discovery import build
# from sklearn import metrics
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import train_test_split
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from tabulate import tabulate
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
# # Load YouTube watch history from a JSON file
# with open('/Users/laurenlanda/Desktop/Takeout/YouTube and YouTube Music/history/watch-history.json', 'r') as file:
#     watch_history = json.load(file)
#
# # Extract relevant information
# video_texts = [video['title'] + ' ' + video.get('description', '') for video in watch_history]
# video_urls = [video['titleUrl'] for video in watch_history]
#
# # Fetch video categories for each video URL
# video_categories = [fetch_video_categories(api_key, url) for url in video_urls]
#
# # Train a simple classifier using video categories
# X_train, X_test, y_train, y_test = train_test_split(video_texts, video_categories, test_size=0.2, random_state=42)
#
#
#
# def preprocess_text(text):
#     words = word_tokenize(text)
#     words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
#     return ' '.join(words)
#
#
# video_texts = [preprocess_text(text) for text in video_texts]
#
# # Limit the number of videos to match the number of categories
# num_videos_to_use = len(video_categories)
# video_texts = video_texts[:num_videos_to_use]
#
# # Create a list of (video_text, category) tuples
# video_data = [(video_texts[i], video_categories[i]) for i in range(len(video_texts))]
#
# # Split the data into training and testing sets
# train_data, test_data = train_test_split(video_data, test_size=0.2, random_state=42, stratify=[category for (_, category) in video_data])
#
# # Extract features and labels for training and testing sets
# X_train, y_train = zip(*train_data)
# X_test, y_test = zip(*test_data)
#
# # Print the distribution of classes in both sets
# print("Training Class Distribution:")
# print({category: y_train.count(category) for category in set(y_train)})
#
# print("\nTesting Class Distribution:")
# print({category: y_test.count(category) for category in set(y_test)})
#
# # Train a simple classifier using video categories
# classifier = make_pipeline(TfidfVectorizer(), MultinomialNB())
# classifier.fit(X_train, y_train)
#
# # Predictions on the test set
# y_pred = classifier.predict(X_test)
#
# # Print the accuracy
# accuracy = metrics.accuracy_score(y_test, y_pred)
# print(f"\nAccuracy on Test Data: {accuracy}")
# # Print the accuracy
# accuracy = metrics.accuracy_score(y_test, y_pred)
# print(f"Accuracy on Test Data: {accuracy}")
# # Define the output directory and file path
# output_directory = 'src/out'
# output_file_path = os.path.join(output_directory, 'genreResults.txt')
#
# # Create the output directory if it doesn't exist
# os.makedirs(output_directory, exist_ok=True)
#
# # Predict the next video category for each video in the dataset
# with open(output_file_path, 'w') as output_file:
#     # Predict the next video category for each video in the dataset
#     predictions = []
#
#     for video_id, current_category in video_categories.items():
#         # Convert video_id to integer
#         video_id = int(video_id)
#
#         if video_id < len(video_texts):
#             current_title = video_texts[video_id]
#
#             # Preprocess the current title
#             current_title = preprocess_text(current_title)
#
#             # Make prediction on the current title
#             predicted_category_prob = classifier.predict_proba([current_title])[0]
#             predicted_category = classifier.classes_[predicted_category_prob.argmax()]
#             probability_percentage = predicted_category_prob.max() * 100
#
#             # Store predictions in a list
#             predictions.append({
#                 'current_category': current_category,
#                 'predicted_category': predicted_category,
#                 'probability_percentage': probability_percentage,
#                 'views': video_id  # Replace this with the actual number of views
#             })
#
#     # Sort predictions based on probability in descending order
#     sorted_predictions = sorted(predictions, key=lambda x: x['probability_percentage'], reverse=True)
#
#     # Write the sorted predictions to the output file
#     with open(output_file_path, 'w') as output_file:
#         for prediction in sorted_predictions:
#             output_line = (f"Given that the current genre of the video is '{prediction['current_category']}', "
#                            f"the likelihood that the next video is '{prediction['predicted_category']}' is {prediction['probability_percentage']:.2f}%\n")
#             output_file.write(output_line)
#
#         # Print chart with predicted and previous categories
#         chart_data = [(prediction['current_category'], prediction['predicted_category']) for prediction in
#                       sorted_predictions]
#
#         output_file.write("\nChart:\n")
#         chart_headers = ["Previous", "Predicted"]
#         chart_str = tabulate(chart_data, headers=chart_headers, tablefmt="pretty")
#         output_file.write(chart_str)
#
#         # Find the most and least watched categories
#         most_watched_category = max(sorted_predictions, key=lambda x: x['views'])['predicted_category']
#         least_watched_category = min(sorted_predictions, key=lambda x: x['views'])['predicted_category']
#
#         output_file.write(f"\nMost Watched Category: {most_watched_category}\n")
#         output_file.write(f"Least Watched Category: {least_watched_category}\n")

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
api_key = 'AIzaSyBB__ugeyjrQxGceHF1S6o0PIqBoOfWD4E'  # Replace with your actual YouTube API key


# Fetch video categories using YouTube API
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

            # You might want to map category_id to category name using another API call
            # For simplicity, I'm returning the category_id here
            return category_id
    # Handle the case where the split didn't provide a valid video ID
    return None


# Load YouTube watch history from a JSON file
with open('/Users/laurenlanda/Desktop/Takeout/YouTube and YouTube Music/history/watch-history.json', 'r') as file:
    watch_history = json.load(file)
# Use only the first 50 entries for testing
watch_history_subset = watch_history[:50]

# Extract relevant information from the subset
video_texts = [video['title'] + ' ' + video.get('description', '') for video in watch_history_subset]
video_urls = [video.get('titleUrl', '') for video in watch_history_subset if 'titleUrl' in video]
video_categories = [fetch_video_category(api_key, url) for url in video_urls]

# Train a simple classifier using video categories
X_train, X_test, y_train, y_test = train_test_split(video_texts, video_categories, test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF vectorizer and Naive Bayes classifier
classifier = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the classifier
classifier.fit(X_train, y_train)
category_mapping = {
    1: 'Film & Animation',
    2: 'Autos & Vehicles',
    10: 'Music',
    15: 'Pets & Animals',
    17: 'Sports',
    18: 'Short Movies',
    19: 'Travel & Events',
    20: 'Gaming',
    21: 'Videoblogging',
    22: 'People & Blogs',
    23: 'Comedy',
    24: 'Entertainment',
    25: 'News & Politics',
    26: 'Howto & Style',
    27: 'Education',
    28: 'Science & Technology',
    29: 'Nonprofits & Activism',
    30: 'Movies',
    31: 'Anime/Animation',
    32: 'Action/Adventure',
    33: 'Classics',
    34: 'Comedy',
    35: 'Documentary',
    36: 'Drama',
    37: 'Family',
    38: 'Foreign',
    39: 'Horror',
    40: 'Sci-Fi/Fantasy',
    41: 'Thriller',
    42: 'Shorts',
    43: 'Shows',
    44: 'Trailers'
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

# Write the sorted predictions to the output file
with open(output_file_path, 'w') as output_file:
    # Predict the next video category for each video in the dataset
    predictions = []

    for video_id, current_category_id in enumerate(y_test):
        # Get the current title
        current_title = X_test[video_id]

        # Make prediction on the current title
        predicted_category_id = classifier.predict([current_title])[0]

        # Map category IDs to category names using the globally defined category_mapping
        current_category_name = category_mapping.get(current_category_id, 'Unknown')
        predicted_category_name = category_mapping.get(predicted_category_id, 'Unknown')

        # Store predictions in a list
        predictions.append({
            'current_category': current_category_name,
            'predicted_category': predicted_category_name
        })

    # Write the predictions to the output file
    for prediction in predictions:
        output_line = f"Actual: {prediction['current_category']} | Predicted: {prediction['predicted_category']}\n"
        output_file.write(output_line)

    # Print chart with actual and predicted categories
    chart_data = [(category_mapping.get(current_category_id, 'Unknown'), category_mapping.get(predicted_category_id, 'Unknown')) for current_category_id, predicted_category_id in zip(y_test, classifier.predict(X_test))]

    output_file.write("\nChart:\n")
    chart_headers = ["Actual", "Predicted"]
    chart_str = tabulate(chart_data, headers=chart_headers, tablefmt="pretty")
    output_file.write(chart_str)

print(f"Results written to: {output_file_path}")