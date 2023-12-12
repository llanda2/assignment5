# Assignment 5: Information Retrieval from Real Data
**Westmont College Fall 2023**

**CS 128 Information Retrieval and Big Data**

*Assistant Professor* Mike Ryu (mryu@westmont.edu) 
## Author Information
* **Name**: Lauren Landa
* ***Code Word***: Genuine

## Problem Description
I wanted to analyze my Google Takeout information to find trends, patterns, etc. in the data. Analyzed data and retrieved the data through techniques covered in class such as TF-IDF Vectorization, cloud platforms, Naive Bayes' Classifier, and more. 

## Extra Credit Opprutunity
* Used Youtube API through GCP
  
## Description of the Solution
1.  Import necessary libraries, including json, os, numpy, Google API client for YouTube (googleapiclient), and scikit-learn for machine learning tasks (sklearn)
2. Set up API Key
3. Fetching Video Categories
      * Implement a function fetch_video_category to fetch the category of a YouTube video using its URL
4.  Load in Youtube Watch History
      * Loaded data in from json file which was obtained through Google Takeout
5. Extract Relevant Information
      * Extract relevant information from the watch history subset, including video texts, URLs, and categories using the previously defined         function (Could do more information but for this project just extracted stated information)
6. Random Sampling
      * Perform random sampling on the video texts and categories
7. Training a Classifier
      * Split the data into training and testing sets
      * Create a pipeline with a TF-IDF vectorizer and a Naive Bayes classifier
      * Train the classifier on the training data
8. Updating Category Mapping
      * Update the category_mapping dictionary based on actual category IDs
9. Making Predictions
      * Make predictions on the test set and evaluate the accuracy
10. Generating Output
      * Define an output directory and file path
      * Write the unique predictions to a text file, including current and predicted categories with probabilities
      * Create a chart displaying actual and predicted categories
      * Determine the most and least-watched categories

## References
1. TF-IDF Vectorization: [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
2. YouTube API: [YouTube Data API Documentation](https://developers.google.com/youtube/v3)
3. General YouTube Activity Analysis: [Medium Article](https://olivia-zha.medium.com/youtube-activity-analysis-f8da36b5832)
4. Using YouTube API through Google Cloud Platform: [YouTube Video](https://www.youtube.com/watch?v=TIZRskDMyA4&t=546s)
5. ChatGPT Prompt and Output:
Prompt: give example code for looking at accuracy and a confusion matrix for a train test split classifier
Output: # Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier  # You can replace this with your chosen classifier
import pandas as pd
Assuming you have a DataFrame 'data' with features and labels
Replace this with your actual data loading and preprocessing steps
For example purposes, let's assume your features are in columns 'feature1', 'feature2', etc. and the label is in the 'label' column.
Load your data
Replace 'your_data.csv' with the actual file name or provide the DataFrame directly if using one
data = pd.read_csv('your_data.csv')
Split the data into features and labels
X = data.drop('label', axis=1)  # Features
y = data['label']  # Labels
Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Initialize the classifier (replace RandomForestClassifier with your chosen classifier)
classifier = RandomForestClassifier()
Train the classifier
classifier.fit(X_train, y_train)
Make predictions on the test set
y_pred = classifier.predict(X_test)
Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix:')
print(conf_matrix)
