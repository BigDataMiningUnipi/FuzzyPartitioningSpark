package iet.unipi.bigdatamining.discretization.impl

import iet.unipi.bigdatamining.discretization.configuration.{CandidateSplitStrategy, FilterStrategy}
import iet.unipi.bigdatamining.discretization.impurity.{Impurities, Impurity}
import org.apache.spark.Logging
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * The class build the metadata for both Discretization algorithms
  *
  * @param numFeatures number of features
  * @param numExamples number of examples in the dataset
  * @param numClasses number of class labels
  * @param categoricalFeatures a set that contains the categorical feature indexes
  * @param classMap a map that associate to each class label value an int
  * @param numBins maximum number of bins for each feature
  * @param impurity name of the fuzzy impurity
  * @param candidateSplitStrategy the strategy for retrieve the candidate splits
  * @param minInfoGain minimum information gain threshold that must be true in each node.
  * @param subsamplingFraction ratio of subsampling for computing only the candidate split
  * @param minInstancesPerSubset minimum ratio of instances that must be present in each subset.
  *                                   Basically top-down discretization approaches are not goin
  *
  * @author Armando Segatori
  */
private[discretization] class FilterMetadata(
                                              val numFeatures: Int,
                                              val numExamples: Long,
                                              val numClasses: Int,
                                              val categoricalFeatures: Set[Int],
                                              val classMap: Map[Double, Int],
                                              val numBins: Array[Int],
                                              val numSplits: Array[Int],
                                              val impurity: Impurity,
                                              val candidateSplitStrategy: CandidateSplitStrategy.CandidateSplitStrategy,
                                              val minInfoGain: Double,
                                              val subsamplingFraction: Double,
                                              val minInstancesPerSubset: Long) extends Serializable {

  /**
    * Check if a given feature is categorical
    *
    * @param featureIndex the index of the feature to be checked
    * @return true if the feature is continuous, false otherwise
    */
  def isCategorical(featureIndex: Int): Boolean = categoricalFeatures.contains(featureIndex)

  /**
    * Check if a given feature is continuous
    *
    * @param featureIndex the index of the feature to be checked
    * @return true if the feature is continuous, false otherwise
    */
  def isContinuous(featureIndex: Int): Boolean = !isCategorical(featureIndex)

  /**
    * Check if a dataset has continuous features
    *
    * @return true if in the dataset there is at least one continuous features, false otherwise
    */
  def hasContinuousFeatures: Boolean = Range(0, numFeatures).exists(isContinuous(_))

}

/**
  * Object that implement some useful method for creating metadata of Discretization algorithm
  */
private[discretization] object FilterMetadata extends Logging {

  /**
    * The method builds the metadata for the problem starting from the input dataset and
    * the strategy used for the discretization algorithm
    *
    * @param input the distributed dataset
    * @param strategy the strategy used by the discretization algorithm
    *
    * @return metadata of the problem
    */
  def buildMetadata(
                     input: RDD[LabeledPoint],
                     strategy: FilterStrategy): FilterMetadata = {

    log.info("Building Metadata")

    val numFeatures = input.map(_.features.size).take(1).headOption.getOrElse{
      throw new IllegalArgumentException(s"Filter requires size of input RDD > 0, " +
        s"but was given by empty one.")
    }

    if (numFeatures != strategy.numFeatures)
      throw new IllegalArgumentException(s"Filter number of features mismatch!\n"+
        s"Filter found number of features from input RDD equal to $numFeatures " +
        s"but was given $strategy.numFeatures.")

    val numExamples = input.count()
    val maxPossibleBins = math.min(strategy.maxBins, numExamples).toInt

    val classMap = input.map(_.label).distinct.collect.sorted.zipWithIndex.toMap
    if (strategy.numClasses != classMap.size)
      throw new IllegalArgumentException(s"Filter number of classes mismatch!\n"+
        s"Filter found number of classes from input RDD equal to $classMap.size " +
        s"but was given $strategy.numClasses.")

    val numBins = Array.fill[Int](numFeatures)(maxPossibleBins)
    val numSplit = numBins.map(_ - 1)
    val impurityType = Impurities.fromString(strategy.impurity)

    val candidateSplitStrategyType = CandidateSplitStrategy.fromString(strategy.candidateSplitStrategy)

    val minInstancesPerSubset = (strategy.minInstancesPerSubsetRatio * numExamples).toLong

    new FilterMetadata(numFeatures, numExamples, strategy.numClasses,
      strategy.categoricalFeatures, classMap, numBins, numSplit, impurityType, candidateSplitStrategyType,
      strategy.minInfoGain, strategy.subsamplingFraction, minInstancesPerSubset)

  }

}