# assignment5
Repository for assignment 5. Using google takeout to look at youtube watch history and analyze the data


#Sources so far
TF-IDF Vectorization: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
youtube api: https://developers.google.com/youtube/v3
General: https://olivia-zha.medium.com/youtube-activity-analysis-f8da36b5832
How to use youtube api through gcp: https://www.youtube.com/watch?v=TIZRskDMyA4&t=546s
Chat GPT Prompts and outputs:

Prompt: give example code for looking at accuracy and a confusion matrix for a train test split classifier
Output: # Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier  # You can replace this with your chosen classifier
import pandas as pd

# Assuming you have a DataFrame 'data' with features and labels
# Replace this with your actual data loading and preprocessing steps
# For example purposes, let's assume your features are in columns 'feature1', 'feature2', etc.
# and the label is in the 'label' column.

# Load your data
# Replace 'your_data.csv' with the actual file name or provide the DataFrame directly if using one
data = pd.read_csv('your_data.csv')

# Split the data into features and labels
X = data.drop('label', axis=1)  # Features
y = data['label']  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifier (replace RandomForestClassifier with your chosen classifier)
classifier = RandomForestClassifier()

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix:')
print(conf_matrix)
