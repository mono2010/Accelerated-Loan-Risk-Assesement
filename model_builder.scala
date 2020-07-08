import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

var  df = sqlContext
  .read
  .format("csv")
  .option("header", "true")
  .option("inferSchema", true) // auto determine data type
  .load("/FileStore/tables/export.csv")

  df = df.withColumnRenamed("default", "label")

  var assembler = new VectorAssembler()
  .setInputCols(Array("loan_amnt", "int_rate", "application_type_Joint App",
       "purpose_credit_card", "purpose_debt_consolidation",
       "purpose_educational", "purpose_home_improvement", "purpose_house",
       "purpose_major_purchase", "purpose_medical", "purpose_moving",
       "purpose_other", "purpose_renewable_energy", "purpose_small_business",
       "purpose_vacation", "purpose_wedding", "grade_B", "grade_C", "grade_D",
       "grade_E", "grade_F", "grade_G"))
  .setOutputCol("features")

df = assembler.transform(df)


var Array(train, test) = df.randomSplit(Array(.8, .2), 42)
var rfc = new RandomForestClassifier()

var rfModel = rfc.fit(train)

var predictions = rfModel.transform(test)
var accuracyEvaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")

  accuracyEvaluator.evaluate(predictions)