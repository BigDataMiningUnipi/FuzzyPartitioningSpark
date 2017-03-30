package iet.unipi.bigdatamining.discretization.impl


/**
  * The class provides some useful methods for helping discretization algorithm
  * in some common operations with Contingency tables, namely the computation of cardinality
  * both crisp and fuzzy
  *
  * @author Armando Segatori
  */
private[discretization] class CardinalityCalculator(
    val ct: ContingencyTable) extends Serializable {

  private def areArgumentsWrong(startIndex: Int, endIndex: Int): Boolean = {
    val numClasses = ct.numClasses
    val binClassCounts = ct.binClassCounts
    (startIndex%numClasses) != 0 || (endIndex%numClasses) != 0 ||
      startIndex > endIndex || startIndex >= binClassCounts.length || endIndex > binClassCounts.length ||
      startIndex < 0 || endIndex < 0
  }
  /**
    * Counts the class distribution from the contingency table
    * between two given indexes.
    *
    * @param startIndex Int the index use as start index (included) for counting the distribution of the
    *            class. The value should be 0 or a multiple of numClasses otherwise an IllegalArgumentException
    *            is thrown
    * @param endIndex Int the index use as end index (excluded) for counting the distribution of the
    *            class Th value should be 0 or a multiple of numClasses otherwise an IllegalArgumentException
    *            is thrown
    * @return Array[Double] the array of size numClasses with the distribution of the classes
    *          from startIndex to endIndex of histogram array
    */
  def labelsCounts(startIndex: Int,
              endIndex: Int): Array[Double] = {

    require(!areArgumentsWrong(startIndex, endIndex),
      s"Start and end values should be between the size of the " +
        s"contingency table, a multiple of numClasses and end index should be greater than the " +
        s"start index. This error can occur when given invalid indexes (such as NaN) are provided. " +
        s"Start Index: $startIndex, End Index: $endIndex")

    val counts = Array.fill[Double](ct.numClasses)(0D)
    Range(startIndex, endIndex).foreach(index => counts(index % ct.numClasses) += ct.binClassCounts(index).toDouble)

    counts
  }

  /**
    * Counts the number of instances from the contingency table between two given indexes.
    *
    * @param startIndex Int the index use as start index (included) for counting the distribution of the
    *            class. The value should be 0 or a multiple of numClasses otherwise an IllegalArgumentException
    *            is thrown
    * @param endIndex Int the index use as end index (excluded) for counting the distribution of the
    *            class Th value should be 0 or a multiple of numClasses otherwise an IllegalArgumentException
    *            is thrown
    * @return Array[Double] the array of size numClasses with the distribution of the classes
    *          from startIndex to endIndex of histogram array
    */
  def count(startIndex: Int,
            endIndex: Int): Long = {

    require(!areArgumentsWrong(startIndex, endIndex),
      s"Start and end values should be between the size of the " +
        s"contingency table, a multiple of numClasses and end index should be greater than the " +
        s"start index. This error can occur when given invalid indexes (such as NaN) are provided. " +
        s"Start Index: $startIndex, End Index: $endIndex")

    ct.binClassCounts.slice(startIndex, endIndex).sum

  }

  /**
    * Compute the fuzzy cardinality of each class label for each fuzzy set.
    * In this case, the fuzzy partition is defined by two triangular fuzzy sets where
    * the cores of the leftmost and rightmost ones are defined upon minimum and maximum points
    * of the subset, namely startIndex and endIndex respectively.
    */
  def fuzzyCardinalityOnTwoFuzzySetsFuzzyPartition(
                                                    startIndex: Int,
                                                    endIndex: Int): Array[Array[Double]] = {

    require(!areArgumentsWrong(startIndex, endIndex),
      s"Start and end values should be between the size of the " +
        s"contingency table, a multiple of numClasses and end index should be greater than the " +
        s"start index. This error can occur when given invalid indexes (such as NaN) are provided." +
        s"Start Index: $startIndex, End Index: $endIndex")

    val s0s1 = Array.fill[Array[Double]](2)(Array.fill[Double](ct.numClasses)(0D))
    if((endIndex-startIndex) != ct.numClasses) { // If true we are referring to the same bins and computation makes no sense
      val min = ct.featureSplits(startIndex / ct.numClasses)
      val diff = ct.featureSplits((endIndex / ct.numClasses) - 1) - min // (max - min) cache for speeding up the membership degree computation
      ct.featureSplits
        .zipWithIndex
        .slice(startIndex/ct.numClasses, endIndex/ct.numClasses)
        .foreach { case (xi, i) =>
          val uS1 = (xi - min) / diff
          Range(0, ct.numClasses).foreach { j =>
            s0s1(0)(j) += ct.binClassCounts(i * ct.numClasses + j) * (1 - uS1) // because is strong
            s0s1(1)(j) += ct.binClassCounts(i * ct.numClasses + j) * uS1
          }
        }
    }
    s0s1
  }

  /**
    * Compute the fuzzy cardinality of each class label to each fuzzy set.
    * In this case, the fuzzy partition is defined by three triangular fuzzy set where
    * the cores of the leftmost and rightmost ones are defined upon minimum and maximum points
    * of the subset (identified by the variable startIndex and endIndex) and the core of middle one is
    * defined upon indexOfCandidateSplit point
    */
  def fuzzyCardinalityOnThreeFuzzySetsFuzzyPartition(
                                                      indexOfMiddleFuzzySets: Int,
                                                      startIndex: Int,
                                                      endIndex: Int): Array[Array[Double]] = {

    require(!areArgumentsWrong(startIndex, endIndex),
      s"Start and end values should be between the size of the " +
        s"contingency table, a multiple of numClasses and end index should be greater than the " +
        s"start index. This error can occur when given invalid indexes (such as NaN) are provided. " +
        s"Start Index: $startIndex, End Index: $endIndex")

    require(indexOfMiddleFuzzySets%ct.numClasses == 0 && indexOfMiddleFuzzySets > startIndex
      && indexOfMiddleFuzzySets < endIndex, s"IndexOfMiddleFuzzySets should be between the start and end indexes " +
      s"or a multiple of numClasses. This error can occur when given invalid indexes (such as NaN) are provided. " +
      s"Start Index: $startIndex, End Index: $endIndex, Index of the Middle Fuzzy Set: $indexOfMiddleFuzzySets")

    /*
     * Matrix of |numFuzzySet| x |numClasses|
     * index 0 -> fuzzy set 0 (s0)
     * index 1 -> fuzzy set 1 (s1)
     * index 2 -> fuzzy set 2 (s2)
     */
    val s0s1s2 = Array.fill[Array[Double]](3)(Array.fill[Double](ct.numClasses)(0D))
    val min = ct.featureSplits(startIndex / ct.numClasses)
    val peak = ct.featureSplits(indexOfMiddleFuzzySets / ct.numClasses - 1)
    // (peak - min) cache for speeding up the membership degree computation
    // Note that it can be 0. In that case the fuzzy set in the middle is placed in the same position
    // of one of the other two. The overall cardinality will be NaN
    var diff = peak - min
    // Calculate cardinality for s0 and the left half of s1 (uSi is equal to uS1)
    Range(startIndex / ct.numClasses, indexOfMiddleFuzzySets / ct.numClasses).foreach { i =>
      val xi = ct.featureSplits(i)
      val uS1 = (xi - min) / diff
      Range(0, ct.numClasses).foreach { j =>
        s0s1s2(1)(j) += ct.binClassCounts(i * ct.numClasses + j) * uS1
        s0s1s2(0)(j) += ct.binClassCounts(i * ct.numClasses + j) * (1 - uS1)
      }
    }

    diff = ct.featureSplits(endIndex / ct.numClasses - 1) - peak // (max - peak) cache for speeding up the membership degree computation
    // Calculate cardinality for s2 and the right half of s1 (uSi is equal to uS2)
    Range(indexOfMiddleFuzzySets / ct.numClasses, endIndex / ct.numClasses).foreach { i =>
      val xi = ct.featureSplits(i)
      val uS2 = (xi - peak) / diff
      Range(0, ct.numClasses).foreach { j =>
        s0s1s2(1)(j) += ct.binClassCounts(i * ct.numClasses + j) * (1 - uS2)
        s0s1s2(2)(j) += ct.binClassCounts(i * ct.numClasses + j) * uS2
      }
    }
    s0s1s2
  }

}
