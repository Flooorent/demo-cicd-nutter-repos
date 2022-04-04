# Databricks notebook source
from datetime import datetime

from feature_engineering import add_acidity_feature
from inference import make_predictions
from mlflow_utils import register_best_model
from training import train_rf

# COMMAND ----------

experiment_version = "v4"
experiment_name = f"/Users/florent.moiny@databricks.com/demos/cicd/using-nutter/experiments/{experiment_version}"
model_name = "demo-flo-cicd-before-refacto"

input_data_path = "/databricks-datasets/wine-quality/winequality-red.csv"

# COMMAND ----------

# DBTITLE 1,Read training data
data = spark.read.option("header", "true").option("sep", ";").csv(input_data_path)
display(data)

# COMMAND ----------

# DBTITLE 1,Feature Engineering
data = add_acidity_feature(data)

# COMMAND ----------

# DBTITLE 1,Training
max_evals = 10
num_parallel_trials = 4

now = datetime.now()
parent_run_name = now.strftime("%Y%m%d-%H%M")

train_rf(data, experiment_name, parent_run_name, max_evals, num_parallel_trials)

# COMMAND ----------

# DBTITLE 1,Register best model as new production model
metric = "mse"
register_best_model(model_name, experiment_name, parent_run_name, metric)

# COMMAND ----------

# DBTITLE 1,Predictions
data_to_predict = spark.read.option("header", "true").option("sep", ";").csv(input_data_path).drop("quality")
data_to_predict = add_acidity_feature(data_to_predict)

preds = make_predictions(data_to_predict, model_name, spark)
display(preds)

# COMMAND ----------


