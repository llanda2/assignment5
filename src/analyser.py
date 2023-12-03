from pyspark.sql import SparkSession
from google.cloud import storage
from io import BytesIO

# Set up a Spark session
spark = SparkSession.builder.appName("GCSExample").getOrCreate()
# GCS bucket and object details
bucket_name = "laurenbucketyoutube"
object_name = "watch-history.json"


# Function to read JSON data from GCS
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
