# Fuzzy Partitioning Spark

This project implements a *Fuzzy Partitioning* (FP) built upon Apache Spark framework for generating strong fuzzy partitions from big data.

Publication:
A. Segatori, F. Marcelloni, W. Pedrycz, **"On Distributed Fuzzy Decision Trees for Big Data"**, *IEEE Transactions on Fuzzy Systems* 
DOI: <a href="https://doi.org/10.1109/TFUZZ.2016.2646746">10.1109/TFUZZ.2016.2646746</a>

## Main Features
Fuzzy classification or regression algorithms usually require that a fuzzy partition has been already defined upon each continuous attribute. For this reason, continuous attributes are usually discretized by optimizing purposely-defined indexes 

This code exploits the classical Fayyad's discretizer based on Minimum Description Length Principle, extending the algorithm by employing fuzzy information gain based on fuzzy entropy for generating strong triangular fuzzy partition on each continuous features. 

The algorithm has been tested by employing several real-world publicly available big datasets, such as the well-known *Susy* and *Higgs* datasets, evaluating the behavior of the algorithm in terms of model complexity (number of fuzzy sets defined in each fuzzy partition), execution time, scalability (i.e. varying the number of cores and increasing dataset size). Moreover, the algorithm has been exploited also by different fuzzy classification algorithms to compare the results in terms of accuracy of such methods with the ones achieved by state-of-the-art algorithms. 

For more details, please read PDF in the *doc* folder or download the original paper available at the following <a href="https://doi.org/10.1109/TFUZZ.2016.2646746">link</a>.

## How to use
 
### Packaging the application
Clone the repository and run the following command:
```sh
mvn clean package
```
Use in your own application the *unipi-fuzzy-partitioning-1.0.jar* jar located in the *target* folder.
Check next section to see how to run the Fuzzy Partitioning algorithm from your code.

### Examples
 
The examples below show how to run FP to define strong fuzzy partitioning by exploiting triangular fuzzy sets.

Configuration Parameters are:
- **numFeatures**: an *int* to store the number of features in the datasets 
- **numClasses** : an *int* to store the number of class labels. It can take values {*0*, ..., *numClasses - 1*}.
- **categoricalFeatures** (default value *Empty Set*): a set that contains the index of each categorical feature.
- **impurity** (default value *FuzzyEntropy*): a *string* to store the impurity measure. The only accepted value is *FuzzyEntropy*
- **maxBins** (default value *1000*): an *int* to store the maximum number of bins used for computing the candidate splits 
- **candidateSplitStrategy** (default value *EquiFreqPerPartition*): a string to store the strategy used for generating the candidate splits. Accepted values are *EquiFreqPerPartition* (default) and *EquiWidth*
- **minInfoGain** (default value *0.00001*): a *double* to store the minimum information gain threshold that must be true in each iteration of the algorithm.
- **subsamplingFraction** (default value *1*): a double to store the ratio of subsampling (if 1 all dataset is considered)
- **minInstancesPerSubsetRatio** (default value *0.0001*): an *int* to store the minimum number of examples that each subset inspected in each iteration of the algorithm must contain

Fuzzy Partitioning supports both Scala and Java. Here, how to defined a Fuzzy Partition using the same inputs of the ones employed in the paper. 

#### Scala
```scala

import org.apache.spark.mllib.util.MLUtils

import iet.unipi.bigdatamining.discretization.FuzzyPartitioning
import iet.unipi.bigdatamining.discretization.model.FuzzyPartitioningModel

// Load and parse the data file.
val rdd = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")

// Set parameters of Fuzzy Partitioing (same of the ones employed in the paper).
//  Empty categoricalFeaturesInfo indicates all features are continuous.
val numFeatures = 4
val numClasses = 3
val categoricalFeatures = Set(1,2)
val impurity = "FuzzyEntropy"
val maxBins = 1000
val candidateSplitStrategy = "EquiFreqPerPartition"
val minInfoGain = 0.000001
val subsamplingFraction = 1D
val minInstancesPerSubsetRatio = 0.0001D

// Run Fuzzy Partitioinng
val fpModel = FuzzyPartitioning.discretize(rdd, numFeatures, numClasses, categoricalFeatures, impurity,
      maxBins, candidateSplitStrategy, minInfoGain, subsamplingFraction, minInstancesPerSubsetRatio)

// Print model complexity
println(s"Totoal number of Fuzzy Sets: ${fpModel.numFuzzySets}") 
println(s"Average number of Fuzzy Sets: ${fpModel.averageFuzzySets}") 
println(s"Number of discarded features: ${fpModel.discardedFeature}") 
println(s"Number of fuzzy sets of the feature with the highest number of fuzzy sets: ${fpModel.max._2}") 
println(s"Number of fuzzy sets if the feature with the lowest number of fuzzy sets (discarded features are not taken in considiration): ${fpModel.min._2}")

``` 

#### Java
```java

import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

import iet.unipi.bigdatamining.classification.tree.FuzzyBinaryDecisionTree
import iet.unipi.bigdatamining.classification.tree.model.FuzzyDecisionTreeModel

SparkConf sparkConf = new SparkConf().setAppName("JavaFuzzyBinaryDecisionTreeExample");
JavaSparkContext jsc = new JavaSparkContext(sparkConf);

// Load and parse the data file.
String datapath = "data/mllib/sample_libsvm_data.txt";
JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();

// Set parameters of Fuzzy Partitioning (same of the ones employed in the paper).
//  Empty categoricalFeaturesInfo indicates all features are continuous.
int numFeatures = 4
int numClasses = 3
Set<Integer> categoricalFeatures = new HashSet<Integer>(1,2)
String impurity = "FuzzyEntropy"
int maxBins = 1000
String candidateSplitStrategy = "EquiFreqPerPartition"
double minInfoGain = 0.000001
double subsamplingFraction = 1D
double minInstancesPerSubsetRatio = 0.0001

// Train a FuzzyBinaryDecisionTree model.
final FuzzyPartitioningModel fpModel = FuzzyPartitioning.discretize(rdd, 
    numFeatures, numClasses, categoricalFeatures, impurity,
    maxBins, candidateSplitStrategy, minInfoGain, subsamplingFraction, 
    minInstancesPerSubsetRatio)
        
// Print model complexity
System.out.println("Totoal number of Fuzzy Sets: " + fpModel.numFuzzySets) 
System.out.println("Average number of Fuzzy Sets: " + fpModel.averageFuzzySets) 
System.out.println("Number of discarded features: " + fpModel.discardedFeature") 
System.out.println("Number of fuzzy sets of the feature with the highest number of fuzzy sets: " + fpModel.max._2) 
System.out.println("Number of fuzzy sets if the feature with the lowest number of fuzzy sets (discarded features are not taken in considiration): " + fpModel.min._2)

``` 


## Unsupported Features
- Sparse Vectors
- Saving/Loading of Models
- ML package


## Contributors

- <a href="https://it.linkedin.com/in/armandosegatori">Armando Segatori</a> (main contributor and maintainer)
- <a href="http://www.iet.unipi.it/f.marcelloni/">Francesco Marcelloni</a>
- <a href="http://www.ece.ualberta.ca/~pedrycz/">Witold Pedrycz</a>


## References

[1] A. Segatori, F. Marcelloni, W. Pedrycz, **"On Distributed Fuzzy Decision Trees for Big Data"**, *IEEE Transactions on Fuzzy Systems*

**Please cite the above work in your manuscript if you plan to use this code**:

###### Plain Text:
```
A. Segatori; F. Marcelloni; W. Pedrycz, "On Distributed Fuzzy Decision Trees for Big Data," in IEEE Transactions on Fuzzy Systems , vol.PP, no.99, pp.1-1
doi: 10.1109/TFUZZ.2016.2646746
```

###### BibTeX
```
@ARTICLE{7803561, 
    author={A. Segatori and F. Marcelloni and W. Pedrycz}, 
    journal={IEEE Transactions on Fuzzy Systems}, 
    title={On Distributed Fuzzy Decision Trees for Big Data}, 
    year={2017}, 
    volume={PP}, 
    number={99}, 
    pages={1-1}, 
    doi={10.1109/TFUZZ.2016.2646746}, 
    ISSN={1063-6706}, 
    month={},}
```


Are you looking for a way for generating *Fuzzy Decision Trees* for Big Data? Check out the **Fuzzy Decision Tree** code available <a href="https://github.com/BigDataMiningUnipi/FuzzyDecisionTreeSpark">here</a>.
