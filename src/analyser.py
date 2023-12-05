import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
__author__ = "Lauren Landa"
__credits__ = ["Lauren Landa"]
__license__ = "MIT"

# Load your YouTube watch history from Google Takeout
with open('/Users/laurenlanda/Desktop/Takeout/YouTube and YouTube Music/history/watch-history.json', 'r') as file:
    watch_history = json.load(file)

# Extract features and labels
titles = [entry['title'] for entry in watch_history]
genres = [entry.get('description', '') for entry in watch_history]  # Using description as a substitute for genre

# Encode labels (genres) into numerical values
label_encoder = LabelEncoder()
encoded_genres = label_encoder.fit_transform(genres)

# Create a mapping of encoded labels to original labels
label_mapping = dict(zip(encoded_genres, genres))

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(titles)

# Prepare the features and labels for training
X_train, X_test, y_train, y_test = train_test_split(X[:-1], encoded_genres[1:], test_size=0.2, random_state=42)

# Create and train a decision tree classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Make predictions on the next video genre
next_video_title = titles[-1]
next_video_title_vectorized = vectorizer.transform([next_video_title])
predicted_genre_encoded = classifier.predict(next_video_title_vectorized)[0]
predicted_genre = label_mapping[predicted_genre_encoded]

# Print the predicted genre
print(f"The predicted genre for the next video ('{next_video_title}') is: {predicted_genre}")