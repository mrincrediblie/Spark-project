package com.example

import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import java.nio.file.{Files, Paths, StandardCopyOption}
import java.io.IOException

object SparkApp {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Spark Recruitment Challenge")
      .master("local[*]")
      .getOrCreate()

    // Paths to the CSV files located in the main directory
    val reviewsFilePath = "google-play-store-apps/googleplaystore_user_reviews.csv"
    val appsFilePath = "google-play-store-apps/googleplaystore.csv"
    val bestAppsOutputPath = "google-play-store-apps/output/best_apps"
    val cleanedOutputPath = "google-play-store-apps/output/googleplaystore_cleaned"
    val metricsOutputPath = "google-play-store-apps/output/googleplaystore_metrics"

    // Delete the outputs directory if it existse
    def deleteDirectoryIfExists(outputPath: String): Unit = {
      try {
        val path = Paths.get(outputPath)
        if (Files.exists(path)) {
          Files.walk(path)
            .sorted(java.util.Comparator.reverseOrder())
            .forEach(Files.delete)
        }
      } catch {
        case e: IOException => println(s"Erro ao deletar o diretório: $e")
      }
    }

    deleteDirectoryIfExists(bestAppsOutputPath)
    deleteDirectoryIfExists(cleanedOutputPath)
    deleteDirectoryIfExists(metricsOutputPath)


    // Task 1: Calculate the average Sentiment_Polarity by App
    val reviewsDF = spark.read.option("header", "true").csv(reviewsFilePath)
    val reviewsDFWithDouble = reviewsDF.withColumn("Sentiment_Polarity", col("Sentiment_Polarity").cast("double"))
    val averageSentimentDF = reviewsDFWithDouble.groupBy("App")
      .agg(avg("Sentiment_Polarity").alias("Average_Sentiment_Polarity"))
      .na.fill(0)

    averageSentimentDF.show()
    
    // Task 2: Get all Apps with a "Rating" greater or equal to 4.0 sorted in descending order and save as best_apps
    val appsDF = spark.read.option("header", "true").csv(appsFilePath)

    // Check for null or invalid values in the Rating column
    val cleanedAppsDF = appsDF.filter(col("Rating").isNotNull && col("Rating") =!= "NaN" && col("Rating") =!= "")

    val bestAppsDF = cleanedAppsDF.filter(col("Rating").cast("double") >= 4.0)
      .orderBy(col("Rating").cast("double").desc)

    bestAppsDF.coalesce(1).write.option("header", "true").option("delimiter", "§").csv(bestAppsOutputPath)
    bestAppsDF.show()  


    // Task 3: Create a DataFrame with the specified structure 
    val df_3 = appsDF.groupBy("App")
      .agg(
        collect_set("Category").alias("Categories"),
        first("Rating").cast("double").alias("Rating"),
        first("Reviews").cast("long").alias("Reviews"),
        first("Size").alias("Size"),
        first("Installs").alias("Installs"),
        first("Type").alias("Type"),
        (first("Price").cast("double") * 0.9).alias("Price"), // Convert to euros
        first("Content Rating").alias("Content_Rating"),
        first("Genres").alias("Genres"),
        first("Last Updated").alias("Last_Updated"),
        first("Current Ver").alias("Current_Version"),
        first("Android Ver").alias("Minimum_Android_Version")
      )
      .withColumn("Categories", concat_ws(";", col("Categories")))
      .withColumn("Size", regexp_replace(col("Size"), "M", "").cast("double"))
      .na.fill(Map("Rating" -> 0, "Reviews" -> 0, "Size" -> 0, "Price" -> 0))
    df_3.show()
    
    
    // Task 4: Merge DataFrames and save as parquet with gzip compression
    val cleanedDF = df_3.join(averageSentimentDF, Seq("App"), "left")
    //cleanedDF.show()
    cleanedDF.write.option("compression", "gzip").parquet(cleanedOutputPath)
    

    // Task 5: Create df_4 DataFrame and save as parquet
    val finalWithSentimentDF = df_3.join(averageSentimentDF, Seq("App"), "left")

    val df_4 = finalWithSentimentDF.groupBy("Genres")
      .agg(
        count("*").alias("Count"),
        avg("Rating").alias("Average_Rating"),
        avg("Average_Sentiment_Polarity").alias("Average_Sentiment_Polarity")
      )
    //df_4.show()
    df_4.write.option("compression", "gzip").parquet(metricsOutputPath)
    
    spark.stop()
  }
}
