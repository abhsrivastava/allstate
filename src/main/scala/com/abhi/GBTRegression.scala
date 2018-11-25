package com.abhi

import org.apache.spark.sql._
import org.apache.spark.ml._
import org.apache.spark.ml.regression._
import org.apache.spark.ml.evaluation._
import org.apache.spark.ml.tuning._
import org.apache.spark.mllib.evaluation._

object GBTRegression extends App {
    val spark = SparkSessionHelper.getSession()
    import spark.implicits._
    
    // define hyper parameters
    val NumTrees = Seq(5, 10, 15)
    val MaxBins = Seq(5, 7, 9)
    val NumFolds = 10
    val MaxIter : Seq[Int] = Seq(10)
    val MaxDepth : Seq[Int] = Seq(10)

    val model = new GBTRegressor()
                    .setFeaturesCol("features")
                    .setLabelCol("label")

    val pipeline = new Pipeline()
                    .setStages((Preprocessing.stringIndexerStages :+ Preprocessing.assembler) :+ model)

    val paramGrid = new ParamGridBuilder()
                        .addGrid(model.maxIter, MaxIter)
                        .addGrid(model.maxDepth, MaxDepth)
                        .addGrid(model.maxBins, MaxBins)
                        .build()

    val cv = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(new RegressionEvaluator)
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(NumFolds)

    val cvModel = cv.fit(Preprocessing.trainingData)

    val trainPredictionsAndLabels = cvModel
                                        .transform(Preprocessing.trainingData)
                                        .select("label", "prediction")
                                        .map { case Row(label: Double, prediction: Double) => (label, prediction) }
                                        .rdd

    val validPredictionsAndLabels = cvModel
                                        .transform(Preprocessing.validationData)
                                        .select("label", "prediction")
                                        .map { case Row(label: Double, prediction: Double) => (label, prediction) }
                                        .rdd        


    val trainRegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels) 
    val validRegressionMetrics = new RegressionMetrics(validPredictionsAndLabels)    
    val bestModel = cvModel
                        .bestModel
                        .asInstanceOf[PipelineModel]

    val featureImportances = bestModel
                                .stages
                                .last.asInstanceOf[GBTRegressionModel]
                                .featureImportances.toArray

    val FI_to_List_sorted = featureImportances
                                .toList
                                .sorted
                                .toArray    

val output = "=====================================================================\n" +
        s"Param trainSample: ${Preprocessing.train}\n" +                    
        s"Param testSample: ${Preprocessing.test}\n" +
        s"TrainingData count: ${Preprocessing.trainingData.count}\n" +
        s"ValidationData count: ${Preprocessing.validationData.count}\n" +
        s"TestData count: ${Preprocessing.test.count}\n" +
        "=====================================================================\n" +
        s"Param maxIter = ${MaxIter.mkString(",")}\n" +
        s"Param maxDepth = ${MaxDepth.mkString(",")}\n" +
        s"Param numFolds = ${NumFolds}\n" +
        "=====================================================================\n" +
        s"Training data MSE = ${trainRegressionMetrics.meanSquaredError}n" +
        s"Training data RMSE = ${trainRegressionMetrics.rootMeanSquaredError}\n" +
        s"Training data R-squared = ${trainRegressionMetrics.r2}\n" +
        s"Training data MAE = ${trainRegressionMetrics.meanAbsoluteError}\n" +
        s"Training data Explained variance = ${trainRegressionMetrics.explainedVariance}\n" +
        "\n=====================================================================\n" +
        s"Validation data MSE = ${validRegressionMetrics.meanSquaredError}\n" +
        s"Validation data RMSE = ${validRegressionMetrics.rootMeanSquaredError}\n" +
        s"Validation data R-squared = ${validRegressionMetrics.r2}\n" +
        s"Validation data MAE = ${validRegressionMetrics.meanAbsoluteError}\n" +
        s"Validation data Explained variance = ${validRegressionMetrics.explainedVariance}\n" +
        "=====================================================================\n" +
        s"CV params explained: ${cvModel.explainParams}\n" +
        s"GBT params explained: ${bestModel.stages.last.asInstanceOf[GBTRegressionModel].explainParams}\n" +
        s"GBT features importances: ${Preprocessing.featureCols.zip(FI_to_List_sorted).map(t => s"t${t._1} = ${t._2}").mkString("\n")}\n" +
        "=====================================================================\n"

    println("Run prediction over test dataset")
    cvModel.transform(Preprocessing.test)
        .select("id", "prediction")
        .withColumnRenamed("prediction", "loss")
        .coalesce(1)
        .write.format("com.databricks.spark.csv")
        .option("header", "true")
        .save("output/result_GBT.csv")        
}