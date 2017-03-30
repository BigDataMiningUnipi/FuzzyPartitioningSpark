package iet.unipi.bigdatamining.discretization.configuration

/**
  * Object to figure out the Split Strategy enum from an input string.
  * Only accepted name: "binary_split", "multi_split".
  */

/**
  * Candidate Splitting Strategy enum. Two strategies:
  *   (1) EquiFreqPerPartition: each partition of the RDD is split according to the equi-frequency strategy,
  *       i.e. each bin contains the same number of points
  *   (2) EquiWidth: the universe of each feature is split according to equi-width strategy,
  *       i.e. each bin has the same width
  *
  * @author Armando Segatori
  */
private[discretization] object CandidateSplitStrategy extends Enumeration {
  type CandidateSplitStrategy = Value
  val EquiFreqPerPartition, EquiWidth = Value

  /**
    * Method to figure out which Candidate Splitting Strategy enum must be selected from an input string.
    * Only accepted name: "equifreqperpartition", "equiwidth".
    *
    * @param name of the candidate split strategy
    * @return a CandidateSplitStrategy enum.
    *         EquiFreqPerPartition in case name equal to equifreqperpartition (not case-sensitive)
    *         EquiWidth in case name equal to equiwidth (not case-sensitive)
    *         [[java.lang.IllegalArgumentException]] in other all other cases.
    */
  def fromString(name: String): CandidateSplitStrategy = name.toLowerCase match {
    case "equifreqperpartition" => CandidateSplitStrategy.EquiFreqPerPartition
    case "equiwidth" => CandidateSplitStrategy.EquiWidth
    case _ => throw new IllegalArgumentException(s"Did not recognize candidate split strategy name: $name")
  }
}
