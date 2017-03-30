package iet.unipi.bigdatamining.discretization

import java.math.RoundingMode
import java.text.DecimalFormat

import iet.unipi.bigdatamining.discretization.configuration.{CandidateSplitStrategy, FilterStrategy}
import iet.unipi.bigdatamining.discretization.impl._
import iet.unipi.bigdatamining.discretization.impurity.Entropy
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import com.holdenkarau.spark.testing.SharedSparkContext
import org.apache.spark.SparkContext
import org.junit.runner.RunWith
import org.scalatest.Matchers._
import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner

import scala.io.Source

/**
  * @author Armando Segatori
  */
@RunWith(classOf[JUnitRunner])
class FuzzyPartitioningSuite extends FunSuite with SharedSparkContext {

  //******************************************************************
  //*      1. Test for building metadata and util methods
  //******************************************************************

  //Test 1.1
  test("Test 1.1: Fuzzy Partitioning with continuous features; testing metadata") {
    val data = FuzzyPartitioningSuite.generateContinuousDataPoints
    data.length should be(1000)
    val rdd = sc.parallelize(data)
    val strategy = new FilterStrategy(
      numFeatures = 2,
      numClasses = 1,
      categoricalFeatures = Set.empty[Int],
      impurity = "FuzzyEntropy",
      maxBins = 10,
      candidateSplitStrategy = "EquiFreqPerPartition",
      minInfoGain = 0.000001,
      subsamplingFraction = 1D,
      minInstancesPerSubsetRatio = 0D
    )
    val metadata = FilterMetadata.buildMetadata(rdd, strategy)
    // Test all properties and methods
    metadata.numFeatures should be(2)
    metadata.numClasses should be(1)
    metadata.categoricalFeatures should be(Set.empty[Int])
    metadata.classMap(1D) should be(0)
    metadata.impurity should be(Entropy)
    metadata.numBins.foreach(_ should be(10))
    metadata.numSplits.foreach(_ should be(9))
    metadata.candidateSplitStrategy should be(CandidateSplitStrategy.EquiFreqPerPartition)
    metadata.minInfoGain should be(0.000001)
    metadata.subsamplingFraction should be(1D)
    metadata.minInstancesPerSubset should be(0)
    metadata.isCategorical(0) should be(false)
    metadata.isCategorical(1) should be(false)
    metadata.isContinuous(0) should be(true)
    metadata.isContinuous(1) should be(true)
    metadata.hasContinuousFeatures should be(true)
  }

  //Test 1.2
  test("Test 1.2: Fuzzy Partitioning with categorical features; testing metadata") {
    val data = FuzzyPartitioningSuite.generateCategoricalDataPoints
    data.length should be(1000)
    val rdd = sc.parallelize(data)
    val strategy = new FilterStrategy(
      numFeatures = 2,
      numClasses = 2,
      categoricalFeatures = Set(0, 1),
      impurity = "FuzzyEntropy",
      maxBins = 100,
      candidateSplitStrategy = "EquiFreqPerPartition",
      minInfoGain = 0.000001,
      subsamplingFraction = 1D,
      minInstancesPerSubsetRatio = 0D
    )
    val metadata = FilterMetadata.buildMetadata(rdd, strategy)
    // Test all properties and methods
    metadata.numFeatures should be(2)
    metadata.numClasses should be(2)
    metadata.categoricalFeatures should be (Set(0, 1))
    metadata.classMap should be (Map(0D -> 0, 1D -> 1))
    metadata.impurity should be(Entropy)
    metadata.numBins.foreach(_ should be(100))
    metadata.numSplits.foreach(_ should be(99))
    metadata.candidateSplitStrategy should be(CandidateSplitStrategy.EquiFreqPerPartition)
    metadata.minInfoGain should be(0.000001)
    metadata.subsamplingFraction should be(1D)
    metadata.minInstancesPerSubset should be(0)
    metadata.isCategorical(0) should be(true)
    metadata.isCategorical(1) should be(true)
    metadata.isContinuous(0) should be(false)
    metadata.isContinuous(1) should be(false)
    metadata.hasContinuousFeatures should be(false)
  }

  //Test 1.3
  test("Test 1.3: Fuzzy Partitioning with both continuous and categorical features; testing metadata") {
    val data = FuzzyPartitioningSuite.generateContinuousDataPointsForMulticlass
    data.length should be(3001)
    val rdd = sc.parallelize(data)
    val strategy = new FilterStrategy(
      numFeatures = 2,
      numClasses = 2,
      categoricalFeatures = Set(0),
      impurity = "FuzzyEntropy",
      maxBins = 100,
      candidateSplitStrategy = "EquiFreqPerPartition",
      minInfoGain = 0.000001,
      subsamplingFraction = 0.9,
      minInstancesPerSubsetRatio = 0.5
    )
    val metadata = FilterMetadata.buildMetadata(rdd, strategy)
    // Test all properties and methods
    metadata.numFeatures should be(2)
    metadata.numClasses should be(2)
    metadata.categoricalFeatures should be (Set(0))
    metadata.classMap should be (Map(1D -> 0, 2D -> 1))
    metadata.impurity should be(Entropy)
    metadata.numBins.foreach(_ should be(100))
    metadata.numSplits.foreach(_ should be(99))
    metadata.candidateSplitStrategy should be(CandidateSplitStrategy.EquiFreqPerPartition)
    metadata.minInfoGain should be(0.000001)
    metadata.subsamplingFraction should be(0.9)
    metadata.minInstancesPerSubset should be(1500)
    metadata.isCategorical(0) should be(true)
    metadata.isCategorical(1) should be(false)
    metadata.isContinuous(0) should be(false)
    metadata.isContinuous(1) should be(true)
    metadata.hasContinuousFeatures should be(true)
  }


  //Test 1.4
  test("Test 1.4: CandidateSplitFeature -> Equi-Frequency method with no duplicated values") {
    val data = FuzzyPartitioningSuite.generateContinuousDataPoints
    data.length should be(1000)
    val numSplits = 99 // 100 bins
    val numDataPerBin = data.length / (numSplits + 1) // number of points in each bin

    // Retrieve the list
    val projectedData0 = data.map(_.features(0))
    val projectedData1 = data.map(_.features(1))
    // Get split values for both features
    val result0 = CandidateSplitFeature.findEquiFrequecySplits(projectedData0, numSplits)
    val result1 = CandidateSplitFeature.findEquiFrequecySplits(projectedData1, numSplits)

    // Compute the expected result
    val expectedResult = projectedData0.min +:
      Range(1, 100).map(i => (projectedData0(i*numDataPerBin) + projectedData0(i*numDataPerBin-1))/2).toArray :+
      projectedData0.max

    result0 should be (expectedResult)
    result0 should be (result1)

    val result2 = CandidateSplitFeature.findEquiFrequecySplits(projectedData0, 9)
    result2 should be (Array(0D, 99.5, 199.5, 299.5, 399.5, 499.5, 599.5, 699.5, 799.5, 899.5, 999D))

    // Edge cases
    //  (i) Empty data
    val emptyData = Array.empty[Double]
    val emptyResult =  CandidateSplitFeature.findEquiFrequecySplits(emptyData, numSplits)
    emptyResult should be (Array.empty[Double])

    // (ii) Zero Splits
    val zeroSplit = CandidateSplitFeature.findEquiFrequecySplits(projectedData1, 0)
    val expectedResultZeroSplit = Array(projectedData1.min, projectedData1.max)
    zeroSplit should be (expectedResultZeroSplit)

  }

  //Test 1.5
  test("Test 1.5: CandidateSplitFeature -> Equi-Width method with no duplicated values") {
    val data = FuzzyPartitioningSuite.generateContinuousDataPointsForMulticlass
    data.length should be(3001)
    val numSplits = 9 // 10 bins

    // Retrieve the list
    val projectedData0 = data.map(_.features(0)).toList
    val projectedData1 = data.map(_.features(1)).toList
    // Get split values for both features
    val result0 = CandidateSplitFeature.findEquiWidthSplits(projectedData0.min, projectedData0.max, numSplits)
    val result1 = CandidateSplitFeature.findEquiWidthSplits(projectedData1.min, projectedData1.max, numSplits)

    result0 should be (Array(2D, 2D))
    result1 should be (Array(0D, 300D, 600D, 900D, 1200D, 1500D, 1800D, 2100D, 2400D, 2700D, 3000D))

    // Edge cases
    // (i) Zero Splits
    val zeroSplit = CandidateSplitFeature.findEquiWidthSplits(projectedData1.min, projectedData1.max, 0)
    val expectedResultZeroSplit = Array(projectedData1.min, projectedData1.max)
    zeroSplit should be (expectedResultZeroSplit)

    a [IllegalArgumentException] should be thrownBy{
      CandidateSplitFeature.findEquiWidthSplits(0D, 10D, -1)
    }

    a [IllegalArgumentException] should be thrownBy{
      CandidateSplitFeature.findEquiWidthSplits(5D, 0D, 10)
    }

  }

  //Test 1.6
  test("Test 1.6: CandidateSplitFeature -> Equi-Frequency method with duplicated values") {
    val data = FuzzyPartitioningSuite.generateContinuousDataPointsWithDuplicates
    data.length should be(100)

    // Retrieve the list
    val projectedData0 = data.map(_.features(0))
    // Get split values for first feature
    val result0 = CandidateSplitFeature.findEquiFrequecySplits(projectedData0, 19) // 20 bins (5 points per bin)

    result0 should be (Array(0D, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 29.5, 54.5, 59.5, 64.5, 69.5, 74.5, 79.5, 84.5, 89.5, 94.5, 99D))

    val result1 = CandidateSplitFeature.findEquiFrequecySplits(projectedData0, 9) // 10 bins (10 points per bin)
    result1 should be (Array(0D, 1.5, 3.5, 5.5, 7.5, 29.5, 59.5, 69.5, 79.5, 89.5, 99D))

    a [IllegalArgumentException] should be thrownBy{
      CandidateSplitFeature.findEquiFrequecySplits(Array.empty[Double], -1)
    }
  }

  // Test 1.7
  test("Test 1.7: MinMaxOnlineSummarizer") {
    val data = FuzzyPartitioningSuite.generateContinuousDataPoints
    data.length should be (1000)

    // Create the online summarizer
    val summarizer1 = new MinMaxOnlineSummarizer
    summarizer1.max should be (Vectors.dense(Array.empty[Double]))
    summarizer1.min should be (Vectors.dense(Array.empty[Double]))

    summarizer1.add(data(0).features)
    summarizer1.max should be (Vectors.dense(0D, 999D))
    summarizer1.min should be (Vectors.dense(0D, 999D))
    summarizer1.add(data(1).features)
    summarizer1.max should be (Vectors.dense(1D, 999D))
    summarizer1.min should be (Vectors.dense(0D, 998D))
    for (i <- 2 until 500){
      summarizer1.add(data(i).features)
    }

    summarizer1.max should be (Vectors.dense(499D, 999D))
    summarizer1.min should be (Vectors.dense(0D, 500D))

    // Create other summarizer
    val summarizer2 = new MinMaxOnlineSummarizer
    summarizer2.add(data(999).features)
    summarizer2.max should be (Vectors.dense(999D, 0D))
    summarizer2.min should be (Vectors.dense(999D, 0D))

    summarizer1.merge(summarizer2)
    summarizer1.max should be (Vectors.dense(999D, 999D))
    summarizer1.min should be (Vectors.dense(0D, 0D))

    for (i <- 0 until 999){
      summarizer1.add(data(0).features)
      summarizer2.add(data(0).features)
    }

    summarizer1.max should be (Vectors.dense(999D, 999D))
    summarizer1.min should be (Vectors.dense(0D, 0D))

    val summarizer3 = new MinMaxOnlineSummarizer
    summarizer3.merge(summarizer1)
    summarizer3.max should be (Vectors.dense(999D, 999D))
    summarizer3.min should be (Vectors.dense(0D, 0D))

    // Feature size equals to 0
    a [RuntimeException] should be thrownBy{
      summarizer1.add(Vectors.dense(Array.empty[Double]))
    }
    // Mismatch feature size when adding new data
    a [RuntimeException] should be thrownBy{
      summarizer1.add(Vectors.dense(0D, 0D, 0D))
    }
    // Mismatch feature size when merging new summarizers
    a [RuntimeException] should be thrownBy{
      val summarizer4 = new MinMaxOnlineSummarizer
      summarizer4.add(Vectors.dense(0D, 0D, 0D))
      summarizer1.merge(summarizer4)
    }
  }


  //Test 1.8
  test("Test 1.8: Contingency Table") {
    val data = FuzzyPartitioningSuite.generateDataForContingencyTable
    data.length should be(100)
    val rdd = sc.parallelize(data)
    val strategy = new FilterStrategy(
      numFeatures = 1,
      numClasses = 2,
      categoricalFeatures = Set.empty[Int],
      impurity = "FuzzyEntropy",
      maxBins = 5,
      candidateSplitStrategy = "EquiFreqPerPartition",
      minInfoGain = 0.000001,
      subsamplingFraction = 0.9,
      minInstancesPerSubsetRatio = 0.5
    )
    val metadata = FilterMetadata.buildMetadata(rdd, strategy)
    val featureSplit = Array[Double](0D, 1.5, 2.5, 19.5, 98.5, 99D)

    // Crete Contingency Table
    val ct1 = new ContingencyTable(metadata, featureSplit)
    ct1.binClassCounts.length should be (14)
    ct1.binClassCounts.sum should be (0D)

    // Add data
    ct1.add(data(0).features(0), data(0).label)
    ct1.binClassCounts should be (Array[Long](1L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L))
    ct1.add(data(1).features(0), data(1).label)
    ct1.binClassCounts should be (Array[Long](1L, 0L, 0L, 1L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L))
    ct1.add(data(2).features(0), data(2).label)
    ct1.binClassCounts should be (Array[Long](1L, 0L, 0L, 1L, 1L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L))
    ct1.add(data(99).features(0), data(99).label)
    ct1.binClassCounts should be (Array[Long](1L, 0L, 0L, 1L, 1L, 0L, 0L, 0L, 0L, 0L, 0L, 1L, 0L, 0L))

    for (i <- 3 until 99) {
      ct1.add(data(i).features(0), data(i).label)
    }

    ct1.binClassCounts should be (Array[Long](1L, 0L, 0L, 1L, 1L, 0L, 8L, 9L, 40L, 39L, 0L, 1L, 0L, 0L))

    // Crete other Contingency Table
    val ct2 = new ContingencyTable(metadata, featureSplit)
    ct1.merge(ct2)
    ct1.binClassCounts should be (Array[Long](1L, 0L, 0L, 1L, 1L, 0L, 8L, 9L, 40L, 39L, 0L, 1L, 0L, 0L))

    // Add data to ct2 and merge them
    ct2.add(data(0).features(0), data(0).label)
    ct1.merge(ct2)
    ct1.binClassCounts should be (Array[Long](2L, 0L, 0L, 1L, 1L, 0L, 8L, 9L, 40L, 39L, 0L, 1L, 0L, 0L))

    // Add all data to ct2
    for (i <- 1 until 100) {
      ct2.add(data(i).features(0), data(i).label)
    }

    // Merge the two contingency tables
    ct1.merge(ct2)
    ct1.binClassCounts should be (Array[Long](3L, 0L, 0L, 2L, 2L, 0L, 16L, 18L, 80L, 78L, 0L, 2L, 0L, 0L))
    ct2.binClassCounts should be (Array[Long](1L, 0L, 0L, 1L, 1L, 0L,  8L,  9L, 40L, 39L, 0L, 1L, 0L, 0L))

    // Unexpected value
    a [RuntimeException] should be thrownBy{
      ct1.add(Double.NaN, 0D)
    }
    // Unexpected label
    a [RuntimeException] should be thrownBy{
      ct1.add(0D, 2D)
    }

    // Mismatch number of classes
    a [RuntimeException] should be thrownBy{
      val strategy = new FilterStrategy(
        numFeatures = 1,
        numClasses = 3,
        categoricalFeatures = Set.empty[Int],
        impurity = "FuzzyEntropy",
        maxBins = 5,
        candidateSplitStrategy = "EquiFreqPerPartition",
        minInfoGain = 0.000001,
        subsamplingFraction = 0.9,
        minInstancesPerSubsetRatio = 0.5
      )
      val metadata = FilterMetadata.buildMetadata(rdd, strategy)
      val featureSplit = Array[Double](0D, 1.5, 2.5, 19.5, 98.5, 99D)

      // Crete Contingency Table
      val ct3 = new ContingencyTable(metadata, featureSplit)
      ct1.merge(ct3)
    }
    // Mismatch splits
    a [RuntimeException] should be thrownBy{
      val strategy = new FilterStrategy(
        numFeatures = 1,
        numClasses = 2,
        categoricalFeatures = Set.empty[Int],
        impurity = "FuzzyEntropy",
        maxBins = 5,
        candidateSplitStrategy = "EquiFreqPerPartition",
        minInfoGain = 0.000001,
        subsamplingFraction = 0.9,
        minInstancesPerSubsetRatio = 0.5
      )
      val metadata = FilterMetadata.buildMetadata(rdd, strategy)
      val featureSplit = Array[Double](0D, 1.5, 3.5, 19.5, 98.5, 99D)

      // Crete Contingency Table
      val ct3 = new ContingencyTable(metadata, featureSplit)
      ct1.merge(ct3)
    }

  }

  //Test 1.9
  test("Test 1.9: CardinalityCalculator - crisp methods") {
    /*
      Generate a Contingency Table with the util
        BinClassCount equals to
        Array[Long](1L, 0L, 0L, 1L, 1L, 0L, 8L, 9L, 40L, 39L, 0L, 1L, 0L, 0L)
     */
    val ct = FuzzyPartitioningSuite.generateContingencyTable(sc)
    val calculator = new CardinalityCalculator(ct)

    // Check labelCounts method results
    calculator.labelsCounts(0, 2) should be (Array(1L, 0L))
    calculator.labelsCounts(12, 14) should be (Array(0L, 0L))
    calculator.labelsCounts(0, 14) should be (Array(50L, 50L))
    calculator.labelsCounts(6, 10) should be (Array(48L, 48L))
    calculator.labelsCounts(6, 12) should be (Array(48L, 49L))

    // Check count method results
    calculator.count(0, 2) should be (1L)
    calculator.count(12, 14) should be (0L)
    calculator.count(0, 14) should be (100L)
    calculator.count(6, 10) should be (96L)
    calculator.count(6, 12) should be (97L)

    //Edge cases
    a [IllegalArgumentException] should be thrownBy{
      calculator.count(1, 14)
    }
    a [IllegalArgumentException] should be thrownBy{
      calculator.count(0, 13)
    }
    a [IllegalArgumentException] should be thrownBy{
      calculator.count(12, 2)
    }
    a [IllegalArgumentException] should be thrownBy{
      calculator.count(0, 104)
    }
    a [IllegalArgumentException] should be thrownBy{
      calculator.count(0, 104)
    }
    a [IllegalArgumentException] should be thrownBy{
      calculator.count(1000, 2000)
    }
    a [IllegalArgumentException] should be thrownBy{
      calculator.count(-2, 2000)
    }
    a [IllegalArgumentException] should be thrownBy{
      calculator.count(-4, -2)
    }
  }

  //Test 1.10
  test("Test 1.10: CardinalityCalculator - fuzzy methods") {
    /*
      Generate a Contingency Table with the util method
        BinClassCount equals to
          Array[Long](1L, 0L, 0L, 1L, 1L, 0L, 8L, 9L, 40L, 39L, 0L, 1L, 0L, 0L)
        Candidate Split equals to
          Array[Double](0D, 1.5, 2.5, 19.5, 98.5, 99D)
     */
    val ct = FuzzyPartitioningSuite.generateContingencyTable(sc)
    val calculator = new CardinalityCalculator(ct)


    // Check fuzzyCardinalityOnTwoFuzzySetsFuzzyPartition method results
    calculator.fuzzyCardinalityOnTwoFuzzySetsFuzzyPartition(0, 2) should be (Array(Array(0D, 0D), Array(0D, 0D)))
    calculator.fuzzyCardinalityOnTwoFuzzySetsFuzzyPartition(0, 4) should be (Array(Array(1D, 0D), Array(0D, 1D)))
    calculator.fuzzyCardinalityOnTwoFuzzySetsFuzzyPartition(0, 6) should be (Array(Array(1D, 0.4), Array(1D, 0.6)))
    calculator.fuzzyCardinalityOnTwoFuzzySetsFuzzyPartition(8, 12) should be (Array(Array(40D, 39D), Array(0D, 1D)))
    calculator.fuzzyCardinalityOnTwoFuzzySetsFuzzyPartition(10, 12) should be (Array(Array(0D, 0D), Array(0D, 0D)))

    // s0 = 1D 0D - 0D 0.985 - 0.975 0D - 6.424 7.227 -  0.202  0.197 - 0D 0D -->  8.601  8.409
    // s1 = 0D 0D - 0D 0.015 - 0.025 0D - 1.576 1.773 - 39.798 38.803 - 0D 1D --> 46.247 41.591
    calculator.fuzzyCardinalityOnTwoFuzzySetsFuzzyPartition(0, 12)
      .map(x =>
        x.map(e =>
          FuzzyPartitioningSuite.roundValue(e)).toList.toArray
      ) should be (Array(Array(8.601, 8.409), Array(41.399, 41.591)))

    calculator.fuzzyCardinalityOnThreeFuzzySetsFuzzyPartition(4, 0, 12)
      .map(x =>
        x.map(e =>
          FuzzyPartitioningSuite.roundValue(e)).toList.toArray
      ) should be (Array(Array(1D, 0D), Array(7.718, 8.538), Array(41.282, 41.462)))

    calculator.fuzzyCardinalityOnThreeFuzzySetsFuzzyPartition(6, 0, 12)
      .map(x =>
        x.map(e =>
          FuzzyPartitioningSuite.roundValue(e)).toList.toArray
      ) should be (Array(Array(1D, 0.4), Array(7.798, 8.217), Array(41.202, 41.383)))

    //Additional edge cases
    a [IllegalArgumentException] should be thrownBy{
      calculator.fuzzyCardinalityOnThreeFuzzySetsFuzzyPartition(3, 0, 14)
    }
    a [IllegalArgumentException] should be thrownBy{
      calculator.fuzzyCardinalityOnThreeFuzzySetsFuzzyPartition(2, 4, 14)
    }
    a [IllegalArgumentException] should be thrownBy{
      calculator.fuzzyCardinalityOnThreeFuzzySetsFuzzyPartition(10, 6, 9)
    }

  }

  //Test 1.11
  test("Test 1.11: Fuzzy Partitioning on Iris dataset") {
    val (traArr, tstArr) = FuzzyPartitioningSuite.getIrisFromCsvToLabeledPoint
    traArr.length should be(135)
    tstArr.length should be (15)

    // Run Fuzzy Partitioning
    val rdd = sc.parallelize(traArr, 1)
    val numFeatures = 4
    val numClasses = 3
    val categoricalFeatures = Set.empty[Int]
    val impurity = "FuzzyEntropy"
    val maxBins = 30
    val candidateSplitStrategy = "EquiFreqPerPartition"
    val minInfoGain = 0.000001
    val subsamplingFraction = 1D
    val minInstancesPerSubsetRatio = 0D

    val fpModel = FuzzyPartitioning.discretize(rdd, numFeatures, numClasses, categoricalFeatures, impurity,
      maxBins, candidateSplitStrategy, minInfoGain, subsamplingFraction, minInstancesPerSubsetRatio)

    fpModel.numFuzzySets should be (16)
    fpModel.averageFuzzySets should be (4D)
    fpModel.discardedFeature should be (Set.empty)
    fpModel.max._2 should be (5)
    fpModel.min._2 should be (3)

    fpModel.cores(0) should be (List(4.3, 5.85, 7.9))
    fpModel.cores(3) should be (List(0.1, 1.05, 1.45, 1.95, 2.5))

  }

  //Test 1.12
  test("Test 1.12: Fuzzy Partitioning on Iris dataset") {
    val (traArr, tstArr) = FuzzyPartitioningSuite.getIrisFromCsvToLabeledPoint
    traArr.length should be(135)
    tstArr.length should be (15)

    // Run Fuzzy Partitioning
    val rdd = sc.parallelize(traArr, 1)
    val numFeatures = 4
    val numClasses = 3
    val categoricalFeatures = Set(1,2)
    val impurity = "FuzzyEntropy"
    val maxBins = 30
    val candidateSplitStrategy = "EquiFreqPerPartition"
    val minInfoGain = 0.000001
    val subsamplingFraction = 1D
    val minInstancesPerSubsetRatio = 0D

    val fpModel = FuzzyPartitioning.discretize(rdd, numFeatures, numClasses, categoricalFeatures, impurity,
      maxBins, candidateSplitStrategy, minInfoGain, subsamplingFraction, minInstancesPerSubsetRatio)

    fpModel.numFuzzySets should be (8)
    fpModel.averageFuzzySets should be (4D)
    fpModel.discardedFeature should be (Set.empty)
    fpModel.max._2 should be (5)
    fpModel.min._2 should be (3)

    fpModel.cores(0) should be (List(4.3, 5.85, 7.9))
    fpModel.cores(3) should be (List(0.1, 1.05, 1.45, 1.95, 2.5))

    a [NoSuchElementException] should be thrownBy{
      fpModel.cores(1)
    }

    a [NoSuchElementException] should be thrownBy{
      fpModel.cores(2)
    }
  }

}

object FuzzyPartitioningSuite extends FunSuite with SharedSparkContext {

  def generateContinuousDataPoints: Array[LabeledPoint] = {
    val arr = new Array[LabeledPoint](1000)
    for (i <- 0 until 1000) {
      val lp = new LabeledPoint(1D, Vectors.dense(i.toDouble, 999D - i))
      arr(i) = lp
    }
    arr
  }

  def generateCategoricalDataPoints: Array[LabeledPoint] = {
    val arr = new Array[LabeledPoint](1000)
    for (i <- 0 until 1000) {
      if (i < 600) {
        arr(i) = new LabeledPoint(1D, Vectors.dense(0D, 1D))
      } else {
        arr(i) = new LabeledPoint(0D, Vectors.dense(1D, 0D))
      }
    }
    arr
  }

  def generateContinuousDataPointsWithDuplicates: Array[LabeledPoint] = {
    val arr = new Array[LabeledPoint](100)
    for (i <- 0 until 100) {
      val point = if (i < 50) {
        i.toDouble%10
      } else {
        i.toDouble
      }
      val lp = new LabeledPoint(1D, Vectors.dense(point, 99D - i))
      arr(i) = lp
    }
    arr
  }

  def generateContinuousDataPointsForMulticlass: Array[LabeledPoint] = {
    val arr = new Array[LabeledPoint](3001)
    for (i <- 0 until 3001) {
      if (i < 2000) {
        arr(i) = new LabeledPoint(2D, Vectors.dense(2D, i))
      } else {
        arr(i) = new LabeledPoint(1D, Vectors.dense(2D, i))
      }
    }
    arr
  }

  def generateOrderedLabeledPoints: Array[LabeledPoint] = {
    val arr = new Array[LabeledPoint](1000)
    for (i <- 0 until 1000) {
      val label = if (i < 100) {
        0D
      } else if (i < 500) {
        1D
      } else if (i < 900) {
        0D
      } else {
        1D
      }
      arr(i) = new LabeledPoint(label, Vectors.dense(i.toDouble, 1000D - i))
    }
    arr
  }

  def generateDataForContingencyTable: Array[LabeledPoint] = {
    val arr = new Array[LabeledPoint](100)
    for (i <- 0 until 100) {
      val label = i%2
      if (label == 0) {
        arr(i) = new LabeledPoint(label, Vectors.dense(i.toLong)) // Class Label 0 takes the i value
      } else {
        arr(i) = new LabeledPoint(label, Vectors.dense(i.toLong)) // Class Label 1 takes always 1
      }
    }
    arr
  }

  def generateContingencyTable(sc: SparkContext): ContingencyTable = {
    // Crete Contingency Table (same of Test 1.8)
    val data = FuzzyPartitioningSuite.generateDataForContingencyTable
    data.length should be(100)
    val rdd = sc.parallelize(data)
    val strategy = new FilterStrategy(
      numFeatures = 1,
      numClasses = 2,
      categoricalFeatures = Set.empty[Int],
      impurity = "FuzzyEntropy",
      maxBins = 5,
      candidateSplitStrategy = "EquiFreqPerPartition",
      minInfoGain = 0.000001,
      subsamplingFraction = 0.9,
      minInstancesPerSubsetRatio = 0.5
    )
    val metadata = FilterMetadata.buildMetadata(rdd, strategy)
    val featureSplit = Array[Double](0D, 1.5, 2.5, 19.5, 98.5, 99D)
    val ct = new ContingencyTable(metadata, featureSplit)
    for (i <- 0 until 100) {
      ct.add(data(i).features(0), data(i).label)
    }

    ct
  }


  def getIrisFromCsvToLabeledPoint: (Array[LabeledPoint], Array[LabeledPoint]) = {
    // Header: sepal-length, sepal-width, petal-length, petal-width, class
    val irisCSV = Source.fromURL(getClass.getResource("/iris.csv")).getLines()
    val (traSetCSV, tstSetCSV) = irisCSV.toArray.splitAt(135)
    // Training first 90% and Test last 10% of the overall data
    val trainingSet = traSetCSV.map{ row =>
      val point = row.split(",")
      val features = point.take(4).map(_.toDouble)
      val classLabel = point.last.toDouble
      new LabeledPoint(classLabel, Vectors.dense(features))
    }
    val testSet = tstSetCSV.map{ row =>
      val point = row.split(",")
      val features = point.take(4).map(_.toDouble)
      val classLabel = point.last.toDouble
      new LabeledPoint(classLabel, Vectors.dense(features))
    }

    (trainingSet, testSet)

  }

  private def roundValue(value: Double): Double ={
    val df = new DecimalFormat("#.###")
    df.setRoundingMode(RoundingMode.HALF_UP)
    df.format(value).toDouble
  }


}
