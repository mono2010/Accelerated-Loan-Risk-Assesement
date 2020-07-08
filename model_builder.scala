// this code is meant to be pasted into a Databricks notebook
// the export.csv generated from the Jupyter Notebook is meant to be uploaded directly into Databricks

// import proper libraries

import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

//reading in the uploaded .csv file
var  df = sqlContext
  .read
  .format("csv")
  .option("header", "true")
  .option("inferSchema", true)
  .load("/FileStore/tables/export.csv")

// assigning a target varible
  df = df.withColumnRenamed("default", "label")

// designating features
  var assembler = new VectorAssembler()
  .setInputCols(Array("loan_amnt", "int_rate", "application_type_Joint App",
       "purpose_credit_card", "purpose_debt_consolidation",
       "purpose_educational", "purpose_home_improvement", "purpose_house",
       "purpose_major_purchase", "purpose_medical", "purpose_moving",
       "purpose_other", "purpose_renewable_energy", "purpose_small_business",
       "purpose_vacation", "purpose_wedding", "grade_B", "grade_C", "grade_D",
       "grade_E", "grade_F", "grade_G"))
  .setOutputCol("features")

//assembing the dataframe
df = assembler.transform(df)

// train test split
var Array(train, test) = df.randomSplit(Array(.8, .2), 42)

// instantiating the random forest classifier
var rfc = new RandomForestClassifier()

// fitting the model
var rfModel = rfc.fit(train)

// gathering the predictions
var predictions = rfModel.transform(test)

//evaluating the model
var accuracyEvaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")

  accuracyEvaluator.evaluate(predictions)