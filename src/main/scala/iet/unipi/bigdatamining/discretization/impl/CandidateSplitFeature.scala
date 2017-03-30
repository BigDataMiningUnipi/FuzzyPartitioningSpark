package iet.unipi.bigdatamining.discretization.impl

/**
  * Object with helpful methods for computing the splitting according to different strategies
  *
  * @author Armando Segatori
  */
private[discretization] object CandidateSplitFeature {

  /**
    * Find the splits according to the equi-frequency strategy
    *
    * @param data a list of points on which computing the equi frequency strategy
    * @param numSplits number of needed splits. The parameter affects the number of points that will be in each bin
    * @return an array containing the splits. Note that the array contains two more splits, one for the minimum and one
    *         for the maximum value of the feature.
    */
  def findEquiFrequecySplits(data: Array[Double],
                             numSplits: Int): Array[Double] = {

    if (numSplits < 0)
      throw new IllegalArgumentException(s"Unexpected number of splits values." +
        s"It should be always non negative value but found $numSplits")

    val candidateSplits = {
      if (data.isEmpty) {
        Array.empty[Double]
      }else{
        // Get count for each distinct value
        val valueCountMap = data.foldLeft(Map.empty[Double, Int]) { (m, x) =>
          m + ((x, m.getOrElse(x, 0) + 1))
        }

        // Sort distinct values
        val valueCounts = valueCountMap.toSeq.sortBy(_._1).toArray

        // stride between splits
        val stride = data.length.toDouble / (numSplits + 1)

        // iterate `valueCount` to find splits
        val splitsBuilder = collection.mutable.ArrayBuilder.make[Double]
        splitsBuilder += valueCounts(0)._1 // Add the min of the partition
        var index = 1
        // currentCount: sum of counts of values that have been visited
        var currentCount = valueCounts(0)._2
        // targetCount: target value for `currentCount`.
        // If `currentCount` is closest value to `targetCount`,
        // then current value is a split threshold.
        // After finding a split threshold, `targetCount` is added by stride.
        var targetCount = stride
        while (index < valueCounts.length) {
          val previousCount = currentCount
          currentCount += valueCounts(index)._2
          val previousGap = math.abs(previousCount - targetCount)
          val currentGap = math.abs(currentCount - targetCount)
          // If adding count of current value to currentCount
          // makes the gap between currentCount and targetCount smaller,
          // previous value is a split threshold.
          if (previousGap < currentGap) {
            splitsBuilder += ((valueCounts(index - 1)._1 + valueCounts(index)._1) / 2)
            targetCount += stride
          }
          index += 1
        }

        splitsBuilder += valueCounts(valueCounts.length - 1)._1 // Add the max value of the partition
        splitsBuilder.result
      }
    }

    candidateSplits
  }

  /**
    * Find the splits according to the equi-width strategy.
    * Given the universe of the feature, identified by its minimum and maximum values, and the number of splits
    * the method returns an array containing the values where the splits occurred (including both min and max values).
    * Basically, the method builds the array with (numSplits + 2) elements where
    * each element is spaced of binWidth from neighbors. binWidth is computed according the total number of needed bins.
    *
    * @param min minimum value of the universe of the feature
    * @param max maximum value of the universe of the feature
    * @param numSplits number of needed splits.
    * @return an array containing the splits. Note that the array contains two more splits, one for the minimum and one
    *         for the maximum value of the feature.
    */
  def findEquiWidthSplits(min: Double, max: Double, numSplits: Int): Array[Double] = {
    if (numSplits < 0)
      throw new IllegalArgumentException(s"Unexpected numbero of splits values." +
        s"It should be always non negative value but found $numSplits")

    if (max > min){
      // (numSplits+1) is the number of bins
      val binWidth = (max - min) / (numSplits + 1)
      // (numSplits+2) is the number of total splits (including min and max)
      Range(0, numSplits + 2).map(i => min + binWidth * i).toArray
    }else{
      if (max == min){
        Array(min, max)
      }else{
        throw new IllegalArgumentException(s"Maximum value must be greater than minimum value." +
          s"Provided $max and $min as maximum and minimum values respectively.")
      }
    }
  }

}
