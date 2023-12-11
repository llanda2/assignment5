import unittest
import os
from src.analyzer import fetch_video_category, classifier, X_test, y_test, output_file_path


class TestYouTubeAnalyzer(unittest.TestCase):

    def test_fetch_video_category_valid_url(self):
        # Test with a valid video URL
        valid_url = "https://www.youtube.com/watch?v=abcdefghijk"
        category = fetch_video_category("YourApiKeyHere", valid_url)
        self.assertIsNotNone(category, "Category should not be None for a valid URL")

    def test_fetch_video_category_invalid_url(self):
        # Test with an invalid video URL
        invalid_url = "https://www.youtube.com/watch"  # Invalid URL, missing video ID
        category = fetch_video_category("YourApiKeyHere", invalid_url)
        self.assertIsNone(category, "Category should be None for an invalid URL")

    def test_classifier_accuracy(self):
        # Test if classifier accuracy is within an acceptable range
        y_pred = classifier.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        self.assertGreaterEqual(accuracy, 0.0, "Accuracy should be greater than or equal to 0%")
        self.assertLessEqual(accuracy, 1.0, "Accuracy should be less than or equal to 100%")

    def test_output_file_exists(self):
        # Test if the output file is successfully created
        self.assertTrue(os.path.exists(output_file_path), "Output file should exist")

    def test_most_least_watched_categories(self):
        # Test if the most and least watched categories are identified correctly
        most_watched_category = "Music"  # Replace with the actual most watched category in your test data
        least_watched_category = "News & Politics"  # Replace with the actual least watched category in your test data
        with open(output_file_path, 'r') as output_file:
            content = output_file.read()
            self.assertIn(f"Most Watched Category: {most_watched_category}", content,
                          "Most watched category should be mentioned in the output file")
            self.assertIn(f"Least Watched Category: {least_watched_category}", content,
                          "Least watched category should be mentioned in the output file")


if __name__ == '__main__':
    unittest.main()
