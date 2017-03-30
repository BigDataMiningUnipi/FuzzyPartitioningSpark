package iet.unipi.bigdatamining.discretization

import iet.unipi.bigdatamining.discretization.configuration.CandidateSplitStrategy
import iet.unipi.bigdatamining.discretization.impl.{CandidateSplitFeature, FilterMetadata, MinMaxOnlineSummarizer, XORShiftRandom}

import org.apache.spark.Logging
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.Map

/**
  * Represents a discretization algorithm that discretize the input data.
  * The algorithm performs only an attribute filtering.
  *
  * @author Armando Segatori.
  */
trait SupervisedAttributeFilter extends Serializable with Logging {

  /**
    * Filter the input data using the discretization algorithm.
    *
    * @param data RDD representing the input data to be discretized
    * @return a map where each key contains the index of the feature
    *        and the value contains the corresponding discretization (i.e. an list of cutPoints)
    */
  def filter(data: RDD[LabeledPoint]): Map[Int, List[Double]]

  /**
    * Discretize input data for examples stored in a JavaRDD.
    *
    * @param testData JavaRDD representing data points to be discretized
    * @return a JavaRDD[java.lang.Double] where each entry contains the corresponding prediction
    */
  def filter(testData: JavaRDD[org.apache.spark.mllib.regression.LabeledPoint]): java.util.Map[Int, java.util.List[Double]] =
    filter(testData.rdd)

  /**
    * Finds candidate splits according to the given strategy.
    *
    * @param data RDD of [[org.apache.spark.mllib.regression.LabeledPoint]]
    *              which represents the data on which retrieving the candidate splits.
    * @param metadata the input metadata used by the algorithm
    * @return a map where each key contains the index of the feature
    *        and the value contains the corresponding candidate splits.
    *
    */
  private[discretization] def findCandidateSplits(
                                                   data: RDD[LabeledPoint],
                                                   metadata: FilterMetadata): Map[Int, Array[Double]] = {

    log.info("Computing Candidate Splits")
    if (metadata.hasContinuousFeatures){
      // Sample the input only if there are continuous features
      val sampledInput = if (metadata.subsamplingFraction < 1D){
        data.sample(withReplacement = false, metadata.subsamplingFraction, new XORShiftRandom().nextInt())
      }else{
        data
      }

      // Finds candidate splits according the strategy adopted
      metadata.candidateSplitStrategy match {
        case CandidateSplitStrategy.EquiFreqPerPartition =>
          performEquiFreqPerPartition(sampledInput, metadata)
        case CandidateSplitStrategy.EquiWidth =>
          performEquiWidth(sampledInput, metadata)
        case _ =>
          throw new UnsupportedOperationException("No supported strategy given")

      }

    }else{
      Map.empty[Int, Array[Double]]
    }

  }

  /**
    * Finds candidate splits performing an Approximate Equi-Frequency method.
    * For each partitions of the RDD and for each feature, first it calculates
    * the equi frequency splits and then merge their together.
    *
    * @param data RDD of [[org.apache.spark.mllib.regression.LabeledPoint]]
    *              which represents the data to discretize.
    * @param metadata the input parameters used for the discretization
    * @return a map where each key contains the index of the feature
    *        and the value contains the corresponding candidate splits
    *        (that include also the min and the max of the universe of the feature).
    */
  private def performEquiFreqPerPartition(
                                          data: RDD[LabeledPoint],
                                          metadata: FilterMetadata): Map[Int, Array[Double]] = {

    def calculateSplitPartition(iter: Iterator[LabeledPoint]): Iterator[(Int, Array[Double])] = {

      // Change from row-based array to column-based array (first row contains all data of first feature, and so on)
      val projectedFeaturesData = iter.map(_.features.toArray).toArray.transpose
      // For each continuous feature, computes the equi frequency method.
      projectedFeaturesData.zipWithIndex.map{ case (featureData, featureIndex) =>
        if (metadata.isContinuous(featureIndex))
          (featureIndex, CandidateSplitFeature.findEquiFrequecySplits(featureData, metadata.numSplits(featureIndex)))
        else
          (featureIndex, Array.empty[Double])
      }.toIterator
    }

    // Finds all candidate splits:
    //  (1) mapPartitions finds the candidate splits for each feature in own partitions
    //  (2) reduceByKey merges together all the candidate splits for each feature found in each partition
    data.mapPartitions(calculateSplitPartition, preservesPartitioning = true)
      .reduceByKey((splitsPart, otherSplitPart) => mergeSortedArray(splitsPart, otherSplitPart).distinct)
      .collectAsMap
  }

  /**
    * Finds candidate splits performing an Equi-Width method.
    * For each feature, first it calculates the min and max values and then
    * computes the splits and merge their together.
    *
    * @param data RDD of [[org.apache.spark.mllib.regression.LabeledPoint]]
    *              which represents the data to discretize.
    * @param metadata the input parameters used for the discretization
    * @return an array of (featureIndex - splits) pair where featureIndex
    *     is the index of the feature and splits is an array of sorted
    *     equal width points (that include also the min and the max)
    */
  private def performEquiWidth(
                                data: RDD[LabeledPoint],
                                metadata: FilterMetadata): Map[Int, Array[Double]] = {

    // Calculates the min and max for each feature
    val minMaxSummarizerStat = data.treeAggregate(new MinMaxOnlineSummarizer)(
      (summarizer, row) => summarizer.add(row.features),
      (summarizer1, summarizer2) => summarizer1.merge(summarizer2))

    // For each continuous feature, computes the equi width method.
    Range(0, metadata.numFeatures).map{ featureIndex =>
      if (metadata.isContinuous(featureIndex)){
        (featureIndex, CandidateSplitFeature.findEquiWidthSplits(minMaxSummarizerStat.min(featureIndex),
          minMaxSummarizerStat.max(featureIndex), metadata.numSplits(featureIndex)))
      }else{
        (featureIndex, Array.empty[Double])
      }
    }.toMap
  }

  /**
    * Merges two sorted array keeping the complexity to O(n),
    * where n is the sum of the lengths of the two array.
    *
    * @param a the first array to merge
    * @param b the second array to merge
    * @return the new sorted array. The size of the new array (n)
    * is the sum of the size of array a and b.
    */
  private def mergeSortedArray(
                                a: Array[Double],
                                b: Array[Double]): Array[Double] = {

    val result = new Array[Double](a.length + b.length)
    var i = 0
    var j = 0
    var k = 0
    while (i < a.length && j < b.length) {
      if (a(i) < b(j)) {
        result(k) = a(i)
        i += 1
      }
      else{
        result(k) = b(j)
        j += 1
      }
      k += 1
    }

    while (i < a.length) {
      result(k) = a(i)
      i += 1
      k += 1
    }

    while (j < b.length) {
      result(k) = b(j)
      j += 1
      k += 1
    }

    result
  }

}
