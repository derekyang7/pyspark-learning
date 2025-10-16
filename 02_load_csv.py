from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum

spark = SparkSession.builder \
    .appName("TaxiDataExercise1") \
    .getOrCreate()

# Load CSV
df = spark.read.csv("data/2020_yellow_taxi_trip_data.csv", header=True, inferSchema=True)
# df = spark.read.option("header", True).csv("data/2020_yellow_taxi_trip_data.csv")
df.printSchema()

row_count = df.count()
print(f"Total rows: {row_count}")

df.show(5)
df.take(5)

df.write.mode("overwrite").parquet("data/2020_yellow_taxi_trip_data_parquet")
df_parquet = spark.read.parquet("data/2020_yellow_taxi_trip_data_parquet")

rows_csv = df.count()
cols_csv = len(df.columns)

rows_parquet = df_parquet.count()
cols_parquet = len(df_parquet.columns)

print(f"CSV: {rows_csv} rows, {cols_csv} columns")
print(f"Parquet: {rows_parquet} rows, {cols_parquet} columns")

if rows_csv == rows_parquet and cols_csv == cols_parquet:
    print("✅ Checksum verified — data matches perfectly.")
else:
    print("❌ Mismatch found — investigate differences.")

df_parquet.show(5)

df_parquet.createOrReplaceTempView("yellow_taxi") # temporary view for SQL queries
spark.sql("SELECT COUNT(*) AS total_rows FROM yellow_taxi").show()

spark.sql("""
    SELECT
        passenger_count,
        ROUND(AVG(total_amount), 2) AS avg_total_amount,
        COUNT(*) AS trip_count
    FROM yellow_taxi
    GROUP BY passenger_count
    ORDER BY passenger_count
""").show()

spark.sql("""
    SELECT VendorID, trip_distance, total_amount
    FROM yellow_taxi
    WHERE TRY_CAST(total_amount AS DOUBLE) > 100
    ORDER BY total_amount DESC
    LIMIT 10
""").show()

df_parquet = df_parquet.withColumn("total_amount", col("total_amount").cast("double"))
df_parquet = df_parquet.withColumn("trip_distance", col("trip_distance").cast("double"))
df_parquet.createOrReplaceTempView("yellow_taxi")

spark.sql("""
    SELECT VendorID, trip_distance, total_amount
    FROM yellow_taxi
    WHERE total_amount > 100
    ORDER BY total_amount DESC
    LIMIT 10
""").show()

df_parquet.write.mode("overwrite").saveAsTable("yellow_taxi_permanent") # permanent table for SQL queries in Spark Warehouse
spark.sql("SELECT * FROM yellow_taxi_permanent LIMIT 5").show()

# Cleaning Data
df.select([_sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]).show() # count nulls in each column

df_clean = df.dropna(subset=["passenger_count"]) # drop rows with nulls in passenger_count

df_clean = df.fillna({
    "passenger_count": 0
}) # fill nulls with 0 for passenger_count

df_clean = df_clean.withColumn("total_amount", col("total_amount").cast("double"))
df_clean = df_clean.withColumn("trip_distance", col("trip_distance").cast("double"))

df_filtered = df_clean.filter(
    (col("trip_distance") > 0) &
    (col("total_amount") > 0) &
    (col("trip_distance") < 100) &
    (col("total_amount") < 1000)
)

# Transforming Data
from pyspark.sql.functions import when, round

df_transformed = df_filtered.withColumn(
    "fare_per_mile",
    round(when(col("trip_distance") > 0, col("total_amount") / col("trip_distance")).otherwise(None), 2)
)

df_lazy = df_transformed.filter(col("total_amount") > 100)
print("This line didn’t run anything yet!")  # no computation yet

# Now trigger an action
df_lazy.count()  # ← this actually executes all prior transformations

df_transformed.write.mode("overwrite").parquet("data/2020_yellow_taxi_cleaned_parquet") # write cleaned data to parquet
df_result = spark.read.parquet("data/2020_yellow_taxi_cleaned_parquet")
print("Rows:", df_result.count())
df_result.show(5)

# Grouping and Aggregating
from pyspark.sql.functions import avg, min, max, count

df_transformed.select(
    avg("total_amount").alias("avg_total_amount"),
    min("total_amount").alias("min_total_amount"),
    max("total_amount").alias("max_total_amount"),
    count("*").alias("row_count")
).show()

df_transformed.groupBy("VendorID").agg(
    round(avg("total_amount"), 2).alias("avg_total"),
    round(avg("trip_distance"), 2).alias("avg_distance"),
    count("*").alias("trip_count")
).orderBy("VendorID").show()

df_transformed.groupBy("VendorID", "time_category").agg(
    round(avg("total_amount"), 2).alias("avg_total"),
    count("*").alias("trip_count")
).orderBy("VendorID", "time_category").show()

df_transformed.createOrReplaceTempView("yellow_taxi_cleaned")

spark.sql("""
    SELECT
        VendorID,
        time_category,
        ROUND(AVG(total_amount), 2) AS avg_total,
        COUNT(*) AS trip_count
    FROM yellow_taxi_cleaned
    GROUP BY VendorID, time_category
    ORDER BY VendorID, time_category
""").show()

df_transformed.groupBy("hour_of_day").agg(
    count("*").alias("trip_count"),
    round(avg("total_amount"), 2).alias("avg_total_amount")
).orderBy("hour_of_day").show()

df_summary = df_transformed.groupBy("VendorID").agg(
    count("*").alias("trip_count"),
    round(avg("total_amount"), 2).alias("avg_total")
)

df_summary.write.mode("overwrite").parquet("data/summary_by_vendor")

# Delta Lake
df_transformed.write.format("delta").mode("overwrite").save("data/delta_table")
spark.read.format("delta").load("tmp/delta_table").show()

# Use explain() and .rdd.getNumPartitions() to inspect plans and partitions.
# On Databricks, use the UI Spark UI to view stages and tasks.
# Avoid collect() on large datasets — prefer limit() or write to file for checks.
# Log with Python logging in driver code; use spark.sparkContext.setLogLevel("WARN") to reduce noise.

df_transformed.explain(extended=True)
df_transformed.rdd.getNumPartitions()

df_transformed.limit(10).write.csv("data/2020_yellow_taxi_cleaned_limited.csv")

# Python logging
import logging
logging.basicConfig(level=logging.WARN)
df_transformed.show()

# Spark JVM logging
spark.sparkContext.setLogLevel("WARN")
df_transformed.show()