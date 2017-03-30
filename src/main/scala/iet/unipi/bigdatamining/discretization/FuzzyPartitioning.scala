package iet.unipi.bigdatamining.discretization

import iet.unipi.bigdatamining.discretization.model.FuzzyPartitioningModel
import iet.unipi.bigdatamining.discretization.configuration.FilterStrategy
import iet.unipi.bigdatamining.discretization.impl._
import iet.unipi.bigdatamining.discretization.impurity.Impurity
import org.apache.spark.Logging
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.JavaConverters._
import scala.collection.Map

/**
  * Class implements the FuzzyPartitioning algorithm as described in
  * [[https://doi.org/10.1109/TFUZZ.2016.2646746]].
  *
  * For academic purpose, please cite paper
  *
  * @author Armando Segatori
  */
class FuzzyPartitioning(private val filterStrategy: FilterStrategy)
  extends SupervisedAttributeFilter with Serializable with Logging {

  filterStrategy.assertValid()

  /**
    * If the partitions is strong the partitionCardinality
    * is equal to the frequency of the examples in the partition
    */
  private def calculateWeightedFuzzyImpurity(
                                              partitions: Array[Array[Double]],
                                              partitionCardinality: Double,
                                              impurity: Impurity): Double = {

    partitions.aggregate(0D)((currEntropy, sj) =>
      currEntropy + (impurity.calculate(sj, sj.sum) * sj.sum), // fuzzyEntropy(sj) * |sj|
      (currEntropyI, currEntropyJ) => currEntropyI+currEntropyJ) / partitionCardinality

  }

  /**
    * Checks if split has to be accepted
    */
  private def stopCondition(
                             gain: Double,
                             priorEntropy: Double,
                             bestCounts: Array[Array[Double]],
                             impurity: Impurity): Boolean = {

    // Calculate number of classes on each fuzzy set (si) and the overall subset
    val bestClassCounts = bestCounts.map(si => si.count(_ > 0D))
    val totalClassCounts = bestCounts.transpose.map(_.sum).count(_ > 0D)

    // Calculate total number instances in the subset
    val totalCounts = bestCounts.map(_.sum).sum //Fuzzy Partition must be strong

    // Compute terms for MDLP formula
    val delta = FuzzyPartitioning.log2(math.pow(3, totalClassCounts)-2) - (
        totalClassCounts*priorEntropy -
          {
            bestCounts.zip(bestClassCounts).map( tuple =>
              tuple._2 * impurity.calculate(tuple._1, tuple._1.sum))
            .sum
          }
        )

    gain >= ((FuzzyPartitioning.log2(totalCounts-1) + delta) / totalCounts)
  }

  /**
    * Recursive method for retrieving the best cores given a partition bounded by first and lastPlusOne
    */
  private def getBestCoresPerSubset(
                                     cardinalityCalculator: CardinalityCalculator,
                                     metadata: FilterMetadata,
                                     first: Int,
                                     lastPlusOne: Int,
                                     isFirstTime: Boolean = false): Array[Int] = {

    val labelCounts = cardinalityCalculator.labelsCounts(first, lastPlusOne)
    // Since the partition is strong, the cardinality is equal to the total frequency (s=totalCounts)
    val s = labelCounts.sum

    // Checks number of instances in the set
    if (s >= metadata.minInstancesPerSubset) {
      // Compute the cardinality of the prior entropy (s)
      val s0s1 = if (isFirstTime) {
        Array(labelCounts, Array.fill[Double](metadata.numClasses)(0D))
      }else{
        cardinalityCalculator.fuzzyCardinalityOnTwoFuzzySetsFuzzyPartition(first, lastPlusOne)
      }

      // Fuzzy Entropy of the full set and initialize the best entropy
      val wPriorFuzzyEntropy = calculateWeightedFuzzyImpurity(s0s1, s, metadata.impurity)

      // Find best entropy
      var bestEntropy = wPriorFuzzyEntropy
      var bestS0S1S2 = Array.empty[Array[Double]]
      var bestIndex = -1
      var leftNumInstances = 0D
      var currCoreIndex = first
      while (currCoreIndex < lastPlusOne) {
        // Checks current number of instances
        if ((leftNumInstances > metadata.minInstancesPerSubset)
          && ((s - leftNumInstances) > metadata.minInstancesPerSubset)) {

          // Compute class cardinality according the MF generation strategy (3-Triangular)
          val s0s1s2 = cardinalityCalculator.fuzzyCardinalityOnThreeFuzzySetsFuzzyPartition(currCoreIndex,
            first, lastPlusOne)
          // Calculate conditional Entropy (the fuzzy entropy in case of splits)
          val currentEntropy = calculateWeightedFuzzyImpurity(s0s1s2, s, metadata.impurity)

          // Save best configuration
          if (currentEntropy < bestEntropy) {
            bestEntropy = currentEntropy
            bestS0S1S2 = s0s1s2.clone
            bestIndex = currCoreIndex
          }
        }

        // Update number of instances and split index
        leftNumInstances += cardinalityCalculator.count(currCoreIndex, currCoreIndex + metadata.numClasses)
        currCoreIndex += metadata.numClasses
      }

      // Calculate and checks gain
      val gain = wPriorFuzzyEntropy - bestEntropy

      if (gain > metadata.minInfoGain && stopCondition(gain, wPriorFuzzyEntropy, bestS0S1S2, metadata.impurity)) {
        // Select split points for the left and right subsets
        val left = getBestCoresPerSubset(cardinalityCalculator, metadata, first, bestIndex)
        val right = getBestCoresPerSubset(cardinalityCalculator, metadata, bestIndex, lastPlusOne)

        // Merge cut-points and return them (the returned array is sorted)
        (left :+ bestIndex) ++ right
      } else {
        Array.empty[Int]
      }
    } else {
      Array.empty[Int]
    }
  }
  /**
    * Retrieve the best fuzzy partition for a specific feature.
    *
    * @param fIndex index ofthe feature on which compute the best fuzzy partition
    * @param contingencyTable the contigency table
    * @return a tuple where the first element is the index of the feature and the second the list of cores of the best
    *         fuzzy partition
    */
  private def getBestFuzzyPartition(
                                  fIndex: Int,
                                  contingencyTable: ContingencyTable): (Int, List[Double]) = {

    val coresFuzzySets = {
      val coreIndexes = getBestCoresPerSubset(new CardinalityCalculator(contingencyTable),
        contingencyTable.metadata,
        // Remove last two bins (last one doesn't contain instances
        // and the other one represents the last bin (identified by maximum number in the universe of the feature)
        0, contingencyTable.binClassCounts.length-2*contingencyTable.numClasses,
        isFirstTime = true)
      if (coreIndexes.length != 0){
        val cores = new Array[Double](coreIndexes.length+2)
        cores(0) = contingencyTable.featureSplits(0)
        coreIndexes.indices.foreach { i => val coreIndex = coreIndexes(i)/contingencyTable.numClasses
          if (coreIndex > 0 && coreIndex < contingencyTable.featureSplits.length)
            cores(i+1) = contingencyTable.featureSplits(coreIndex) }
        cores(cores.length-1) = contingencyTable.featureSplits(contingencyTable.featureSplits.length-1)
        cores
      }else{
        Array.empty[Double]
      }
    }

    (fIndex, coresFuzzySets.toList)
  }

  /**
    * Compute the best fuzzy partition for each feature across all possible candidate fuzzy partitions
    *
    * @param data input dataset
    * @param metadata of the FuzzyPartition algorithm
    * @param candidateFuzzySets a map that contains for each feature (key) an array of possible candidate fuzzy partitions
    * @return a map that contains for each feature the best fuzzy partitions, identified by a list of points, where
    *         each point represents the core of the corresponding fuzzy partition.
    */
  private def findBestFuzzyPartitions(
                              data: RDD[LabeledPoint],
                              metadata: FilterMetadata,
                              candidateFuzzySets: Map[Int, Array[Double]]): Map[Int, List[Double]] = {

    log.info("Calculating Best Fuzzy Partitions")

    /**
      * The lpHistogramPartition function computes the partial contigency table for a given
      * partition according the label distribution. The provided histogramFunction
      * determines which bin in the array must be incremented or returns None if there is no bins.
      * Possible histogramFunction are:
      *   (1) basicHistogramFunction
      *   (2) fastHistogramFunction
      */
    def lpHistogramPartition(iter: Iterator[LabeledPoint]): Iterator[(Int, ContingencyTable)] = {

      log.info("Building contingency table for the features")

      val histograms = candidateFuzzySets.flatMap{ case (featureIndex, splits) =>
        if (metadata.isContinuous(featureIndex))
          Some((featureIndex, new ContingencyTable(metadata, candidateFuzzySets(featureIndex))))
        else
          None
      }

      while(iter.hasNext) {
        val lp = iter.next
        histograms.foreach { case (featureIndex, ct) =>
          ct.add(lp.features(featureIndex), lp.label)
        }
      }

      histograms.toIterator

    }

    data.mapPartitions(lpHistogramPartition)
      .reduceByKey((ct, otherCt) => ct.merge(otherCt),
        metadata.numFeatures-metadata.categoricalFeatures.size) // Set number of partitions equal to the number of continuous features
      .map(x => getBestFuzzyPartition(x._1, x._2))
      .collectAsMap()

  }

  /**
    * Filter the input data using the discretization algorithm.
    *
    * @param data RDD representing the input data to be discretized
    * @return a map where each key contains the index of the feature
    *        and the value contains the corresponding discretization (i.e. an list of cutPoints)
    */
  def filter(data: RDD[LabeledPoint]): Map[Int, List[Double]] = {
    // Build the necessary metadata
    val metadata = FilterMetadata.buildMetadata(data, filterStrategy)
    // Find the splits using a sample of the input data (default 'all' data are considered).
    val candidateFuzzySets = findCandidateSplits(data, metadata)
    // Find the best splits for each feature from the candidate split extracted in the previous step
    findBestFuzzyPartitions(data, metadata, candidateFuzzySets)
  }

  /**
    * Entry method of FuzzyPartitioning algorithm
    *
    * @param data input dataset
    * @return a FuzzyPartitioning model that can be used to create new Triangular Fuzzy Partitions
    */
  def run(data: RDD[LabeledPoint]): FuzzyPartitioningModel = {
    // Since we pass on dataset 3 times (1 for building metadata and the other two for filtering)
    // is more convenient persist data for keeping them in main memory
    data.persist
    val coreFuzzySets = filter(data)
    data.unpersist(false)
    // Builds the model and return it
    new FuzzyPartitioningModel(coreFuzzySets)
  }

}

/**
  *  Factory object to discretize data according the Fuzzy Partitioning algorithm
  */
object FuzzyPartitioning extends Logging with Serializable {
  def log2: Double = scala.math.log(2)
  def log2(x: Double): Double = scala.math.log(x)/log2

  /**
    * Method to discretize the feature over an input data according to the '''FuzzyPartitioning''' algorithm.
    * For more detail, please check out [[https://doi.org/10.1109/TFUZZ.2016.2646746]]
    *
    * @param input Training dataset: RDD of [[org.apache.spark.mllib.regression.LabeledPoint]].
    * @param numFeatures number of features
    * @param numClasses number of classes
    * @param categoricalFeatures a set that contains the categorical feature indexes (default empty set, i.e. all features
    *                            are continuous)
    * @param impurity name of the fuzzy impurity (so far only accepted value is 'fuzzyentropy' string)
    * @param maxBins maximum number of bins for each feature (default 1000)
    * @param candidateSplitStrategy the strategy for retrieve the candidate splits (default 'EquiFreqPerPartition' string)
    * @param minInfoGain minimum information gain threshold that must be true in each node (default 0.000001).
    * @param subsamplingFraction ratio of subsampling for computing only the candidate split
    *                            (default 1, i.e. all dataset is considered)
    * @param minInstancesPerSubsetRatio minimum ratio of instances that must be present in each subset.
    *                                   Basically top-down discretization approaches are not going to further split a subset
    *                                   if the number of instances is lower than such a threshold (default 0).
    * @return FilterModel that can be used for filtering the input data.
    */
  def discretize(
                  input: RDD[LabeledPoint],
                  numFeatures: Int,
                  numClasses: Int,
                  categoricalFeatures: Set[Int] = Set.empty[Int],
                  impurity: String = "FuzzyEntropy",
                  maxBins: Int = 1000,
                  candidateSplitStrategy: String = "EquiFreqPerPartition",
                  minInfoGain: Double = 0.000001,
                  subsamplingFraction: Double = 1D,
                  minInstancesPerSubsetRatio: Double = 0D): FuzzyPartitioningModel = {
    val filterStrategy = new FilterStrategy(numFeatures, numClasses, categoricalFeatures, impurity,
      maxBins, candidateSplitStrategy, minInfoGain, subsamplingFraction, minInstancesPerSubsetRatio)
    new FuzzyPartitioning(filterStrategy).run(input)
  }

  /**
    * Java-friendly API for [[iet.unipi.bigdatamining.discretization.FuzzyPartitioning#discretize]].
    */
  def discretize(
                  input: JavaRDD[LabeledPoint],
                  numFeatures: Integer,
                  numClasses: Integer,
                  categoricalFeatures: java.util.Set[java.lang.Integer],
                  impurity: java.lang.String,
                  maxBins: Integer,
                  candidateSplitStrategy: java.lang.String,
                  minInfoGain: java.lang.Double,
                  subsamplingFraction: java.lang.Double,
                  minInstancesPerSubsetRatio: java.lang.Double): FuzzyPartitioningModel = {

    val scalaCategoricalFeatures = categoricalFeatures.asScala.map(x => x.intValue()).toSet
    discretize(input.rdd, numFeatures.toInt, numClasses.toInt, scalaCategoricalFeatures, impurity,
      maxBins.toInt, candidateSplitStrategy, minInfoGain.toDouble,
      subsamplingFraction.toDouble, minInstancesPerSubsetRatio.toDouble)
  }

}
