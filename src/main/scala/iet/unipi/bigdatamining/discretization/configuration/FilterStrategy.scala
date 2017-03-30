package iet.unipi.bigdatamining.discretization.configuration

import scala.beans.BeanProperty

/**
  * The class contains the information about the parameters used for computing the discretization algorithm.
  *
  * @param numFeatures total number of features
  * @param numClasses number of class labels
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
  *
  * @author Armando Segatori
  */
class FilterStrategy(
                      @BeanProperty var numFeatures: Int,
                      @BeanProperty var numClasses: Int,
                      @BeanProperty var categoricalFeatures: Set[Int] = Set.empty[Int],
                      @BeanProperty var impurity: String = "fuzzyentropy",
                      @BeanProperty var maxBins: Int = 1000,
                      @BeanProperty var candidateSplitStrategy: String = "EquiFreqPerPartition",
                      @BeanProperty var minInfoGain: Double = 0.000001,
                      @BeanProperty var subsamplingFraction: Double = 1D,
                      @BeanProperty var minInstancesPerSubsetRatio: Double = 0D) extends Serializable {

  /**
    * Check validity of parameters.
    * Throws exception if invalid.
    */
  private[discretization] def assertValid(): Unit = {
    require(Set("entropy", "fuzzyentropy").contains(impurity.toLowerCase),
      s"Filter Strategy given invalid impurity parameter: $impurity." +
        s"Valid setting are: 'Entropy', 'FuzzyEntropy'")
    require(Set("equifreqperpartition", "equiwidth").contains(candidateSplitStrategy.toLowerCase),
      s"Filter Strategy given invalid candidate split strategy: $candidateSplitStrategy." +
        s"Valid settings are: 'EquiFreqPerPartition', 'EquiWidth'")
    require(subsamplingFraction > 0D && subsamplingFraction <= 1D,
      s"Filter Strategy given invalid subsamplingFraction parameter: $subsamplingFraction." +
        s"The subsamplingFraction parameter must be a value greater than 0 (empty dataset) " +
        s"and less or equal than 1 (all dataset)")
    require(minInstancesPerSubsetRatio >= 0D && minInstancesPerSubsetRatio <= 1D,
      s"Filter Strategy given invalid minInstancesPerSubsetRatio parameter: $minInstancesPerSubsetRatio." +
        s"The minInstancesPerSubsetRatio parameter must be a value greater or equal than 0 (empty dataset) " +
        s"and less or equal than 1 (all dataset)")
  }

  /**
    * Returns a shallow copy of this instance.
    */
  def copy: FilterStrategy = {
    new FilterStrategy(numFeatures, numClasses, categoricalFeatures, impurity, maxBins,
      candidateSplitStrategy, minInfoGain, subsamplingFraction, minInstancesPerSubsetRatio)
  }

}

/**
  * A companion object
  */
object FilterStrategy {

  /**
    * Construct a default set of parameters for [[iet.unipi.bigdatamining.discretization.FuzzyPartitioning]]
    */
  def defaultStrategy: FilterStrategy = {
    new FilterStrategy(numFeatures = -1, numClasses = 2)
  }
}