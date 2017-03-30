package iet.unipi.bigdatamining.discretization.impurity

/**
  * Trait for computing the impurity values in order to calculate the information gain
  *
  * @author Armando Segatori
  */
trait Impurity extends Serializable{

  /**
    * Method for computing the impurity value
    *
    * @param counts array that stores the cardinality per class label
    * @param totalCount total number of instances
    *
    * @return the impurity value
    */
  def calculate(counts: Array[Double], totalCount: Double): Double


}
