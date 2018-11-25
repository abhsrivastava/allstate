package com.abhi

import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression._
import org.apache.spark.ml._

object Preprocessing {
    def isCategory(c: String) : Boolean = c.startsWith("cat") 
    def categoryNewCol(c: String) : String = if (isCategory(c)) s"idx_$c" else c
    def includeCol(c: String) : Boolean = !(c matches "cat(109$|110$|112$|113$|116)")
    def onlyFeatures(c: String) : Boolean = !(c matches "id$|label")

    val spark = SparkSessionHelper.getSession()
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
}