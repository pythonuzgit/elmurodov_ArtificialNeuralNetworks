# Artificial Neural Networks <h1>

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

precision    recall  f1-score   support

           0       0.92      0.80      0.86        15
           1       0.57      0.80      0.67         5

   micro avg       0.80      0.80      0.80        20
   macro avg       0.75      0.80      0.76        20
weighted avg       0.84      0.80      0.81        20
  
  
