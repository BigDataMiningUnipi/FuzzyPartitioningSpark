package iet.unipi.bigdatamining.discretization.impl

import iet.unipi.bigdatamining.discretization.configuration.CandidateSplitStrategy
import org.apache.spark.annotation.Experimental

/**
  *
  * @author Armando Segatori
  */
private[discretization] class ContingencyTable(
    val metadata: FilterMetadata,
    val featureSplits: Array[Double]) extends Serializable {

  val numClasses: Int = metadata.numClasses
  val binClassCounts: Array[Long] = Array.fill[Long]((featureSplits.length+1)*numClasses)(0)

  // Basic histogram function. This works using Java's built in Array
  // binary search. Takes log(size(buckets))
  private def basicSearchIndexFunction(e: Double): Option[Int] = {
    if (e.isNaN){
      None
    } else {
      var index = java.util.Arrays.binarySearch(featureSplits, e)
      // If the location is less than 0 then the insertion point in the array
      // to keep it sorted is -location-1
      if (index < 0) {
        index = -index - 1
      }
      // Exact match to the last element
      Some(index)
    }
  }

  @Experimental
  private def fastSearchIndexFunction(e: Double): Option[Int] = {
    val min = featureSplits.head
    val max = featureSplits.last
    val count = featureSplits.length-1
    // If our input is not a number unless the increment is also NaN then we fail fast
    if (e.isNaN || e < min || e > max) {
      None
    } else {
      // Compute ratio of e's distance along range to total range first, for better precision
      val index = (((e - min) / (max - min)) * count).toInt
      // Should be less than count, but it will equal to count if e == max, in that case
      // it's part of the last end-range-inclusive bin, so return count-1
      Some(math.min(index, count - 1))
    }
  }

  private val searchFunction = metadata.candidateSplitStrategy match {
    case CandidateSplitStrategy.EquiWidth => fastSearchIndexFunction _
    case _ => basicSearchIndexFunction _
  }

  /**
    * Add a new sample to this aggregator, and update the contingency table.
    *
    * @param value to be added into this contingency table.
    * @return This ContingencyTable object.
    */
  def add(value: Double, label: Double): Unit = {
    require(metadata.classMap.get(label).nonEmpty, s"Label mismatch when adding new sample." +
      s" Expecting ${metadata.classMap.keySet} but got $label.")

    searchFunction(value) match {
      case Some(x: Int) => // val index = if (x == 0) 1 else x
        binClassCounts(x * metadata.numClasses + metadata.classMap.get(label).get) += 1L
      case _ => throw new RuntimeException(s"Unexpected value when computing histograms." +
        s" This error can occur when given invalid data values (such as NaN). Value: $value.")

    }

  }

  /**
    * Merge another ContingencyTable, and update the contingency tables.
    * (Note that it's in place merging; as a result, `this` object will be modified.)
    *
    * @param other The other ContingencyTable to be merged.
    * @return This ContingencyTable object.
    */
  def merge(other: ContingencyTable): this.type = {
    require(numClasses == other.numClasses, s"NUmber of classes mismatch when merging with another contingency table. " +
      s"Expecting $numClasses but got ${other.numClasses}.")
    require(featureSplits.sameElements(other.featureSplits), s"Candidate Splits mismatch when merging " +
      s"with another contingency table. Expecting $featureSplits but got ${other.featureSplits}")
    // Merge two contingency tables
    this.binClassCounts.indices.foreach(j => this.binClassCounts(j) += other.binClassCounts(j))
    this
  }

}
