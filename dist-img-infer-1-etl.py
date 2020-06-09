# Databricks notebook source
# MAGIC %md ## 1. ETL images into a Delta table
# MAGIC 
# MAGIC ---
# MAGIC * Use [flowers dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers) hosted under `dbfs:/databricks-datasets`.
# MAGIC * Use binary file data source from Apache Spark to store images in a Spark table.
# MAGIC * Extract image metadata and store them together with image data.
# MAGIC * Use Delta Lake to simplify data management.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Training change

# COMMAND ----------

import io
import numpy as np
import pandas as pd
from pyspark.sql.functions import col, pandas_udf, regexp_extract
from PIL import Image

# COMMAND ----------

# MAGIC %md ### The flowers dataset
# MAGIC 
# MAGIC We use the [flowers dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers) from the TensorFlow team as our example dataset,
# MAGIC which contains flower photos stored under five sub-directories, one per class.
# MAGIC It is hosted under Databricks Datasets for easy access.

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/flower_photos

# COMMAND ----------

# MAGIC %md ### Load images into a DataFrame using binary file data source.
# MAGIC 
# MAGIC Databricks Runtime 5.4 and above support the binary file data source, which reads binary files and converts each file into a single record that contains the raw content and metadata of the file.

# COMMAND ----------

images = spark.read.format("binaryFile") \
  .option("recursiveFileLookup", "true") \
  .option("pathGlobFilter", "*.jpg") \
  .load("/databricks-datasets/flower_photos")

# COMMAND ----------

# MAGIC %md ###Expand the DataFrame with extra metadata columns.
# MAGIC 
# MAGIC We extract some frequently used metadata from `images` DataFrame:
# MAGIC * extract labels from file paths,
# MAGIC * extract image sizes.

# COMMAND ----------

def extract_label(path_col):
  """Extract label from file path using built-in SQL functions."""
  return regexp_extract(path_col, "flower_photos/([^/]+)", 1)

# COMMAND ----------

def extract_size(content):
  """Extract image size from its raw content."""
  image = Image.open(io.BytesIO(content))
  return image.size

# COMMAND ----------

@pandas_udf("width: int, height: int")
def extract_size_udf(content_series):
  sizes = content_series.apply(extract_size)
  return pd.DataFrame(list(sizes))

# COMMAND ----------

df = images.select(
  col("path"),
  col("modificationTime"),
  extract_label(col("path")).alias("label"),
  extract_size_udf(col("content")).alias("size"),
  col("content"))

# COMMAND ----------

display(df.drop("content").limit(5))

# COMMAND ----------

# MAGIC %md ###Save as a Delta table.

# COMMAND ----------

# MAGIC %sql CREATE DATABASE IF NOT EXISTS ml_tmp

# COMMAND ----------

# Image data is already compressed. So we turn off Parquet compression.
spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")
df.write.format("delta").mode("overwrite").saveAsTable("ml_tmp.flowers")

# COMMAND ----------

# MAGIC %md ###Make SQL queries (optional).

# COMMAND ----------

# MAGIC %sql SELECT COUNT(*) FROM ml_tmp.flowers WHERE label = 'daisy'

# COMMAND ----------

# MAGIC %sql SELECT label, COUNT(*) AS cnt FROM ml_tmp.flowers
# MAGIC   WHERE size.width >= 400 AND size.height >= 400
# MAGIC   GROUP BY label ORDER BY cnt