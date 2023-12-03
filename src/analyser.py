import gcsfs
from pyspark.sql import SparkSession
from google.cloud import storage
from io import BytesIO
import json
#Hmmm why no work
#link used: https://stackoverflow.com/questions/58708081/how-to-read-json-file-in-python-code-from-google-cloud-storage-bucket
gcs_file_system = gcsfs.GCSFileSystem(project="AssignmentFive")
gcs_json_path = "gs://laurenbucketyoutube/history/watch-history.json"
with gcs_file_system.open(gcs_json_path) as f:
    json_dict = json.load(f)


# Function to read JSON data from GCS
def read_gcs_data(bucket_name, object_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(object_name)
    content = blob.download_as_text()
    return BytesIO(content.encode())


# Read a sample of data from GCS into a PySpark DataFrame
data = spark.read.json(read_gcs_data(bucket_name, object_name))

# Display a few rows of the DataFrame
data.show()

# Stop the Spark session
spark.stop()
