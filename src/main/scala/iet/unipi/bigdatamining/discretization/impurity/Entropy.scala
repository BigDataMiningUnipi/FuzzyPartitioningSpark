package iet.unipi.bigdatamining.discretization.impurity

/**
  * The class implements the entropy logic (both fuzzy and crisp).
  * It implement the [[iet.unipi.bigdatamining.discretization.impurity.Impurity]] trait.
  *
  * @author Armando Segatori
  */
private[discretization] object Entropy extends Impurity {
  val log2 : Double = scala.math.log(2)
  private def nlnFunc(x: Double): Double = if (x <= 0) 0D else x*scala.math.log(x)

  /**
    * Information calculation for multiclass classification
    * In this class the entropy is used as impurity metrics.
    *
    * @param counts array of double with counts for each class label
    * @param totalCount sum of counts for all labels
    * @return entropy value, or 0 if totalCount lower or equal than 0
    */
  override def calculate(counts: Array[Double], totalCount: Double): Double = {
    if (totalCount > 0D){
      val impurity = counts.foldLeft(0D)(_ + -nlnFunc(_))
      (impurity + nlnFunc(totalCount)) / (totalCount * log2)
    } else 0D

  }

}