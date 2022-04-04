# Databricks notebook source
from pyspark.sql import functions as F

# COMMAND ----------

def add_acidity_feature(df):
  return df.withColumn("total_acidity", F.col("fixed acidity") + F.col("volatile acidity"))
