package com.abhi

import org.apache.spark._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression._
import org.apache.spark.ml._
import org.apache.spark.ml.evaluation._
import org.apache.spark.ml.tuning._
import org.apache.spark.mllib.evaluation._

object AllState extends App {
    def isCategory(c: String) : Boolean = c.startsWith("cat") 
    def categoryNewCol(c: String) : String = if (isCategory(c)) s"idx_$c" else c
    def includeCol(c: String) : Boolean = !(c matches "cat(109$|110$|112$|113$|116)")
    def onlyFeatures(c: String) : Boolean = !(c matches "id$|label")
    // initialize the spark session
    val spark = SparkSession
                    .builder
                    .master("local[*]")
                    .config("spark.sql.warehouse.dir", "/User/data")
                    .appName("AllStateInsurance")
                    .getOrCreate()

    import spark.implicits._

    // load the training and test data
    val test = spark.read.format("csv")
                    .option("sep", ",")
                    .option("inferSchema", "true")
                    .option("header", "true")
                    .load("src/main/resources/test.csv")
                    .na.drop()

    val train = spark.read.format("csv")
                    .option("sep", ",")
                    .option("inferSchema", "true")
                    .option("header", "true")
                    .load("src/main/resources/train.csv")
                    .withColumnRenamed("loss", "label")
                    .na.drop() // drop null data

    val splits = train.randomSplit(Array(.75, .25), 12345L)
    val (trainingData, validationData) = (splits(0), splits(1))

    // identify the columns which should be used as features
    val featureCols = trainingData
                        .columns
                        .filter(onlyFeatures)
                        .filter(includeCol)
                        .map(categoryNewCol)
    
    val stringIndexerStages = trainingData.columns
                                .filter(isCategory)
                                .map{c => 
                                    new StringIndexer()
                                        .setInputCol(c)
                                        .setOutputCol(categoryNewCol(c))
                                        .fit(
                                            train.select(c).union(test.select(c))
                                        )
                                }
    val assembler = new VectorAssembler()
                        .setInputCols(featureCols)
                        .setOutputCol("features")

    // define hyper parameters
    val numFolds = 10
    val MaxIter : Seq[Int] = Seq(1000)
    val RegParam : Seq[Double] = Seq(.001)
    val Tol : Seq[Double] = Seq(1e-6)
    val ElasticNetParam : Seq[Double] = Seq(.001)

    // let us build the model
    val model = new LinearRegression()
                    .setFeaturesCol("features")
                    .setLabelCol("label")

    // let us build the ML pipeline
    val pipeline = new Pipeline()
                    .setStages((stringIndexerStages :+ assembler) :+ model)
    

    val paramGrid = new ParamGridBuilder()
                        .addGrid(model.maxIter, MaxIter)
                        .addGrid(model.regParam, RegParam)
                        .addGrid(model.tol, Tol)
                        .addGrid(model.elasticNetParam, ElasticNetParam)
                        .build()

    val crossValidator = new CrossValidator()
                            .setEstimator(pipeline)
                            .setEvaluator(new RegressionEvaluator)
                            .setEstimatorParamMaps(paramGrid)
                            .setNumFolds(numFolds)

    val cvModel = crossValidator.fit(trainingData)
    val trainPredictionAndLabels = cvModel
                                    .transform(trainingData)
                                    .select("label", "prediction")
                                    .map{case Row(label: Double, prediction: Double) => (label, prediction)}
                                    .rdd
    val trainRegressionMetrics = new RegressionMetrics(trainPredictionAndLabels)

    val validationPredictionAndLabels = cvModel
                                    .transform(trainingData)
                                    .select("label", "prediction")
                                    .map{case Row(label: Double, prediction: Double) => (label, prediction)}
                                    .rdd
    val validationRegressionMetrics = new RegressionMetrics(validationPredictionAndLabels)
    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]

    val results = 
        s"===========================================\n" + 
        s"Training Data Count: ${trainingData.count}\n" +
        s"Validation Data Count: ${validationData.count}\n" +
        s"Test Data Count: ${test.count}\n" +
        s"===========================================\n" + 
        s"Param: Max Iterations: ${MaxIter}\n" +
        s"Param: Max Folds: ${numFolds}\n"
        s"===========================================\n" + 
        s"Train Data MSE: ${trainRegressionMetrics.meanSquaredError}\n" + 
        s"Train Data RMSE: ${trainRegressionMetrics.rootMeanSquaredError}\n" + 
        s"Train Data RSquared: ${trainRegressionMetrics.r2}\n"    
        s"Train Data MAE: ${trainRegressionMetrics.meanAbsoluteError}\n" + 
        s"Train Data Explained Variance: ${trainRegressionMetrics.explainedVariance}\n"
        s"===========================================\n" + 
        s"Validation Data MSE: ${validationRegressionMetrics.meanSquaredError}\n" + 
        s"Validation Data RMSE: ${validationRegressionMetrics.rootMeanSquaredError}\n" + 
        s"Validation Data RSquared: ${validationRegressionMetrics.r2}\n" +     
        s"Validation Data MAE: ${validationRegressionMetrics.meanAbsoluteError}\n" + 
        s"Validation Data Explained Variance: ${validationRegressionMetrics.explainedVariance}\n" + 
        s"===========================================\n" + 
        s"CV Params Explained: ${cvModel.explainParams}\n" +
        s"LR Params Explained: ${bestModel.stages.last.asInstanceOf[LinearRegressionModel].explainParams}\n" + 
        s"===========================================\n"
    
    println(results)     

    // OK so now the actual prediction
    println("Run prediction on the test set")
    cvModel.transform(test)
            .select("id", "prediction")
            .withColumnRenamed("prediction", "loss")
            .coalesce(1) // to get all the predictions in a single csv file
            .write.format("com.databricks.spark.csv")
            .option("header", "true")
            .save("output/result_LR.csv")
            
    spark.stop()

}