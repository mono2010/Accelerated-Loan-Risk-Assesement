import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
import pyspark.ml.evaluation as evals
import pyspark.ml.tuning as tune
df = (spark.read.option("header","true").csv("/FileStore/tables/projects/Accelerated-Loan-Risk-Assessment/loan.csv"))
data = df.select('loan_amnt', 'application_type', 'int_rate', 'purpose', 'grade', 'loan_status')
data = data.withColumn("loan_amnt", data.loan_amnt.cast('integer'))
data = data.withColumn("int_rate", data.int_rate.cast('float'))
data = data.filter('loan_amnt is not NULL and application_type is not NULL and int_rate is not NULL and purpose is not NULL and grade is not NULL and loan_status is not NULL')
app_indexer = StringIndexer(inputCol = 'application_type', outputCol = 'app_index')
app_feature = OneHotEncoder(inputCol = 'app_index', outputCol = 'app')

purpose_indexer = StringIndexer(inputCol = 'purpose', outputCol = 'purpose_index')
purpose_feature = OneHotEncoder(inputCol = 'purpose_index', outputCol = 'purp')

grade_indexer = StringIndexer(inputCol = 'grade', outputCol = 'grade_index')
grade_feature = OneHotEncoder(inputCol = 'grade_index', outputCol = 'grad')


data = data.withColumn("default", data.loan_status == 'Charged Off')
data = data.withColumn("label", data.default.cast('integer'))
assembler = VectorAssembler(inputCols=['loan_amnt', 'int_rate', 'app', 'purp', 'grad'], outputCol='features')
pipe = Pipeline(stages=[app_indexer, app_feature, purpose_indexer, purpose_feature, grade_indexer, grade_feature, assembler])
pipeline = pipe.fit(data).transform(data)
train, test = pipeline.randomSplit([.67, .33])
rfc = RandomForestClassifier()

grid = tune.ParamGridBuilder()
grid = grid.addGrid(rfc.numTrees, np.arange(100, 150))
grid = grid.addGrid(rfc.maxDepth, [0, 5])
grid = grid.build()

evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")

cv = tune.CrossValidator(estimator=rfc, estimatorParamMaps=grid, evaluator=evaluator)
models = cv.fit(train)
MLlib will automatically track trials in MLflow. After your tuning fit() call has completed, view the MLflow UI to see logged runs.