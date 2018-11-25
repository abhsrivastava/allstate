package com.abhi

import org.apache.spark.sql.SparkSession

object SparkSessionHelper {
    def getSession() : SparkSession = {
        // initialize the spark session
        SparkSession
            .builder
            .master("local[*]")
            .config("spark.sql.warehouse.dir", "/User/data")
            .appName("AllStateInsurance")
            .getOrCreate()
    }
}