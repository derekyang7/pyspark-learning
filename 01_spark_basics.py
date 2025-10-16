from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("Basics").getOrCreate()

# create DataFrame
data = [("Alice", 34), ("Bob", 45), ("Cathy", 29)]
df = spark.createDataFrame(data, schema=["name", "age"])
df.show()

# basic transforms
df.select(col("name"), (col("age") + 1).alias("age_plus_one")).show()

# filter & group
df.filter(col("age") > 30).show()
spark.stop()
