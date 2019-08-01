# Artificial Neural Networks using Scikit-Learn in Python - Diabetes Dataset
 
 * Environment: Python 3 and Jupyter Notebook
 * Library: Pandas and NumPy
 * Module: Scikit-Learn
 
 *Understanding Dataset*
 
*There are nine columns in the dataset, which are as follows*
* preg     768 non-null int64
* plas     768 non-null int64
* pres     768 non-null int64
* skin     768 non-null int64
* insu     768 non-null int64
* mass     768 non-null float64
* pedi     768 non-null float64
* age      768 non-null int64
* class    768 non-null object

dtypes: float64(2), int64(6), object(1)
memory usage: 60.0+ KB

*By visualizing the distribution of the eight variables in the dataset.* 
 
  
  
 ![Alt Text](https://github.com/pythonuzgit/elmurodov/blob/master/myfig.png)
   
*From the histogram for age, we can see that most of the data was clloecter from young people, with the most common age group between 20-30 years old. We can also see that the distribution for other concentration is normally disributed.*

*We will build the neural network model using the scikit-learn library's, 'Multi-Layer Perceptron Classifier'. The first parameter 'hidden_layer_sizes' arugument set to three layers 10 nodes each.*

*The above output shows the performance of the model on training data. The accuracy and the f1-score is around 0.81, respectively.*


