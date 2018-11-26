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
import java.io.{PrintWriter, FileOutputStream}

object LinearRegression extends App {

    val spark = SparkSessionHelper.getSession()
    import spark.implicits._

    // define hyper parameters
    val numFolds = 10
    val MaxIter : Seq[Int] = Seq(10)
    val RegParam : Seq[Double] = Seq(.001)
    val Tol : Seq[Double] = Seq(1e-6)
    val ElasticNetParam : Seq[Double] = Seq(.001)

    // let us build the model
    val model = new LinearRegression()
                    .setFeaturesCol("features")
                    .setLabelCol("label")

    // let us build the ML pipeline
    val pipeline = new Pipeline()
                    .setStages((Preprocessing.stringIndexerStages :+ Preprocessing.assembler) :+ model)
    

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

    val cvModel = crossValidator.fit(Preprocessing.trainingData)
    val trainPredictionAndLabels = cvModel
                                    .transform(Preprocessing.trainingData)
                                    .select("label", "prediction")
                                    .map{case Row(label: Double, prediction: Double) => (label, prediction)}
                                    .rdd
    val trainRegressionMetrics = new RegressionMetrics(trainPredictionAndLabels)

    val validationPredictionAndLabels = cvModel
                                    .transform(Preprocessing.validationData)
                                    .select("label", "prediction")
                                    .map{case Row(label: Double, prediction: Double) => (label, prediction)}
                                    .rdd
    val validationRegressionMetrics = new RegressionMetrics(validationPredictionAndLabels)
    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]

    val results = 
        s"===========================================\n" + 
        s"Training Data Count: ${Preprocessing.trainingData.count}\n" +
        s"Validation Data Count: ${Preprocessing.validationData.count}\n" +
        s"Test Data Count: ${Preprocessing.test.count}\n" +
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
    
    new PrintWriter(new FileOutputStream("output/result_LR.txt", false)) { write(results); close}   

    // save the model
    cvModel
        .write
        .overwrite()
        .save("model/LR_model")

    // load the model
    val fittedModel = CrossValidatorModel.load("model/LR_model")
    // OK so now the actual prediction
    println("Run prediction on the test set")
    fittedModel.transform(Preprocessing.test)
            .select("id", "prediction")
            .withColumnRenamed("prediction", "loss")
            .coalesce(1) // to get all the predictions in a single csv file
            .write.format("com.databricks.spark.csv")
            .mode(SaveMode.Overwrite)
            .option("header", "true")
            .save("output/result_LR.csv")
            
    spark.stop()

}