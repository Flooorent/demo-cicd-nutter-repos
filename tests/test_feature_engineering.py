# Databricks notebook source
# install nutter (and all necessary dependencies) on the cluster if you want to trigger tests from the command line
%pip install -r ../requirements.txt

# COMMAND ----------

from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql.types import StructType, DoubleType, IntegerType
from runtime.nutterfixture import NutterFixture, tag

from feature_engineering import add_acidity_feature

# COMMAND ----------

default_timeout = 600


class TestFeatures(NutterFixture):
  def assertion_add_acidity_features(self):
    input_schema = (
      StructType()
        .add("fixed acidity", DoubleType(), True)
        .add("volatile acidity", DoubleType(), True)
        .add("quality", IntegerType(), True)
    )
    
    input_df = spark.createDataFrame(
      [
        [7.4, 0.7, 5],
      ],
      schema=input_schema
    )
    
    actual_df = add_acidity_feature(input_df)
    
    expected_schema = (
      StructType()
        .add("fixed acidity", DoubleType(), True)
        .add("volatile acidity", DoubleType(), True)
        .add("quality", IntegerType(), True)
        .add("total_acidity", DoubleType(), True)
    )
    
    expected_df = spark.createDataFrame(
      [
        [7.4, 0.7, 5, 8.1],
      ],
      schema=expected_schema
    )
    
    assert_df_equality(expected_df, actual_df)


# COMMAND ----------

result = TestFeatures().execute_tests()
print(result.to_string())

is_job = dbutils.notebook.entry_point.getDbutils().notebook().getContext().currentRunId().isDefined()
if is_job:
  result.exit(dbutils)


# COMMAND ----------


