package iet.unipi.bigdatamining.discretization.impurity

/**
  * Factory for Impurity instances
  *
  * @author Armando Segatori
  */
object Impurities {

  /**
    * The method creates an Impurity object from an input string.
    * Only accepted name: "fuzzyentropy" or "entropy". Not case-sensitive
    *
    * @param name of the fuzzy entropy
    * @return a Impurity instance
    *         FEntropy in case name equal to "fuzzyentropy" or "entropy"(not case-sensitive)
    *         [[java.lang.IllegalArgumentException]] in other all other cases.
    */
  def fromString(name: String): Impurity = name.toLowerCase match {
    case "entropy" | "fuzzyentropy" => Entropy
    case _ => throw new IllegalArgumentException(s"Did not recognize Impurity name: $name")
  }
}
