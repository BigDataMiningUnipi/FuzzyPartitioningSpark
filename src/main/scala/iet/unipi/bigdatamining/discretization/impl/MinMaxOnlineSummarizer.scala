package iet.unipi.bigdatamining.discretization.impl

import org.apache.spark.mllib.linalg.{Vector, Vectors}

/**
  * MinMaxOnlineSummarizer computes the minimum and maximum, for samples in sparse or dense vector
  * format in a online fashion.
  *
  * Two MinMaxOnlineSummarizer can be merged together to have a statistical summary of
  * the corresponding joint dataset.
  *
  * @author Armando Segatori
  */
private[discretization] class MinMaxOnlineSummarizer extends Serializable {

  private var n: Int = 0
  private var currMin: Array[Double] = _
  private var currMax: Array[Double] = _

  /**
    * Add a new sample to this summarizer, and update the min max statistical summary.
    *
    * @param sample The sample in dense/sparse vector format to be added into this summarizer.
    * @return This MinMaxStatisticalSummary object.
    */
  def add(sample: Vector): this.type = {
    if (n == 0) {
      require(sample.size > 0, s"Vector should have dimension larger than zero.")
      n = sample.size

      currMax = Array.fill[Double](n)(Double.MinValue)
      currMin = Array.fill[Double](n)(Double.MaxValue)
    }

    require(n == sample.size, s"Dimensions mismatch when adding new sample." +
      s" Expecting $n but got ${sample.size}.")

    val localCurrMax = currMax
    val localCurrMin = currMin
    var i = 0
    while (i < sample.size){
      if (localCurrMax(i) < sample(i)) {
        localCurrMax(i) = sample(i)
      }
      if (localCurrMin(i) > sample(i)) {
        localCurrMin(i) = sample(i)
      }

      i += 1
    }

    this
  }

  /**
    * Merge another MinMaxOnlineSummarizer, and update the statistical summary.
    * (Note that it's in place merging; as a result, `this` object will be modified.)
    *
    * @param other The other MinMaxOnlineSummarizer to be merged.
    * @return This MinMaxOnlineSummarizer object.
    */
  def merge(other: MinMaxOnlineSummarizer): this.type = {
    if (n != 0) {
      require(n == other.n, s"Dimensions mismatch when merging with another summarizer. " +
        s"Expecting $n but got ${other.n}.")
      var i = 0
      while (i < n) {
        // merge max and min
        currMax(i) = math.max(currMax(i), other.currMax(i))
        currMin(i) = math.min(currMin(i), other.currMin(i))
        i += 1
      }
    } else {
      this.n = other.n
      this.currMax = other.currMax.clone()
      this.currMin = other.currMin.clone()
    }

    this
  }

  /**
    * Get the maximum values for each feature
    *
    * @return a vector where each element contains the minimum value for the corresponding feature
    */
  def max: Vector = {
    if (n != 0) {
      Vectors.dense(currMax)
    } else {
      Vectors.dense(Array.empty[Double])
    }
  }

  /**
    * Get the minimum values for each feature
    *
    * @return a vector where each element contains the minimum value for the corresponding feature
    */
  def min: Vector = {
    if (n != 0) {
      Vectors.dense(currMin)
    } else {
      Vectors.dense(Array.empty[Double])
    }
  }

}
