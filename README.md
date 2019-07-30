# Artificial Neural Networks

The first step is to import this dataset into our program. We wil use Python's pandas library.

import pandas as pd

Reading the Data and Performing Basic Data checks

diabet = pd.read_csv('diabetes_csv.csv')


The first step is to import this dataset into our program. We wil use Python's pandas library.

import pandas as pd

Reading the Data and Performing Basic Data checks

diabet = pd.read_csv('diabetes_csv.csv')

To see what this dataset actually looks like, execute the following command.

diabet.head()

	preg 	plas 	pres 	skin 	insu 	mass 	pedi 	age 	class
0 	6 	148 	72 	35 	0 	33.6 	0.627 	50 	tested_positive
1 	1 	85 	66 	29 	0 	26.6 	0.351 	31 	tested_negative
2 	8 	183 	64 	0 	0 	23.3 	0.672 	32 	tested_positive
3 	1 	89 	66 	23 	94 	28.1 	0.167 	21 	tested_negative
4 	0 	137 	40 	35 	168 	43.1 	2.288 	33 	tested_positive

Assign data from first seven columns to X variable

X = diabet.iloc[:, 0:8]

print(X)

     preg  plas  pres  skin  insu  mass   pedi  age
0       6   148    72    35     0  33.6  0.627   50
1       1    85    66    29     0  26.6  0.351   31
2       8   183    64     0     0  23.3  0.672   32
3       1    89    66    23    94  28.1  0.167   21
4       0   137    40    35   168  43.1  2.288   33
5       5   116    74     0     0  25.6  0.201   30
6       3    78    50    32    88  31.0  0.248   26
7      10   115     0     0     0  35.3  0.134   29
8       2   197    70    45   543  30.5  0.158   53
9       8   125    96     0     0   0.0  0.232   54
10      4   110    92     0     0  37.6  0.191   30
11     10   168    74     0     0  38.0  0.537   34
12     10   139    80     0     0  27.1  1.441   57
13      1   189    60    23   846  30.1  0.398   59
14      5   166    72    19   175  25.8  0.587   51
15      7   100     0     0     0  30.0  0.484   32
16      0   118    84    47   230  45.8  0.551   31
17      7   107    74     0     0  29.6  0.254   31
18      1   103    30    38    83  43.3  0.183   33
19      1   115    70    30    96  34.6  0.529   32
20      3   126    88    41   235  39.3  0.704   27
21      8    99    84     0     0  35.4  0.388   50
22      7   196    90     0     0  39.8  0.451   41
23      9   119    80    35     0  29.0  0.263   29
24     11   143    94    33   146  36.6  0.254   51
25     10   125    70    26   115  31.1  0.205   41
26      7   147    76     0     0  39.4  0.257   43
27      1    97    66    15   140  23.2  0.487   22
28     13   145    82    19   110  22.2  0.245   57
29      5   117    92     0     0  34.1  0.337   38
..    ...   ...   ...   ...   ...   ...    ...  ...
738     2    99    60    17   160  36.6  0.453   21
739     1   102    74     0     0  39.5  0.293   42
740    11   120    80    37   150  42.3  0.785   48
741     3   102    44    20    94  30.8  0.400   26
742     1   109    58    18   116  28.5  0.219   22
743     9   140    94     0     0  32.7  0.734   45
744    13   153    88    37   140  40.6  1.174   39
745    12   100    84    33   105  30.0  0.488   46
746     1   147    94    41     0  49.3  0.358   27
747     1    81    74    41    57  46.3  1.096   32
748     3   187    70    22   200  36.4  0.408   36
749     6   162    62     0     0  24.3  0.178   50
750     4   136    70     0     0  31.2  1.182   22
751     1   121    78    39    74  39.0  0.261   28
752     3   108    62    24     0  26.0  0.223   25
753     0   181    88    44   510  43.3  0.222   26
754     8   154    78    32     0  32.4  0.443   45
755     1   128    88    39   110  36.5  1.057   37
756     7   137    90    41     0  32.0  0.391   39
757     0   123    72     0     0  36.3  0.258   52
758     1   106    76     0     0  37.5  0.197   26
759     6   190    92     0     0  35.5  0.278   66
760     2    88    58    26    16  28.4  0.766   22
761     9   170    74    31     0  44.0  0.403   43
762     9    89    62     0     0  22.5  0.142   33
763    10   101    76    48   180  32.9  0.171   63
764     2   122    70    27     0  36.8  0.340   27
765     5   121    72    23   112  26.2  0.245   30
766     1   126    60     0     0  30.1  0.349   47
767     1    93    70    31     0  30.4  0.315   23

[768 rows x 8 columns]

Assign data from first eigth columns to y variable

y = diabet.select_dtypes(include = [object])

#print(y)

y.head()

	class
0 	tested_positive
1 	tested_negative
2 	tested_positive
3 	tested_negative
4 	tested_positive

We have two unique classes "tested_positive" and "tested negative". Lets convert these categorical values to numerical values. We will use Scikit-learn LabelEncoder class.

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

y = y.apply(le.fit_transform)

Train Test Split

We will divide our dataset into training and test splits. The training data will be used to train the neural network and the test data will be used to evaluate the performance of the neural network. This helps with the problem of over-fitting because we are evaluating our neural network on data that it has not seen (i.e. been trained on) before.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 20)

​

Feature Scaling

The following script performs feature scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

​

​

/usr/local/lib/python3.5/dist-packages/sklearn/preprocessing/data.py:625: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
  return self.partial_fit(X, y)

StandardScaler(copy=True, with_mean=True, with_std=True)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

/usr/local/lib/python3.5/dist-packages/ipykernel_launcher.py:1: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
  """Entry point for launching an IPython kernel.
/usr/local/lib/python3.5/dist-packages/ipykernel_launcher.py:2: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
  

Evaluating the Neural Network model

We will build the neural network model using the scikit-learn library's, 'Multi-Layer Perceptron Classifier'. The first parameter 'hidden_layer_sizes' arugument set to three layers 10 nodes each.

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes = (10, 10, 10), max_iter = 1850)

mlp.fit(X_train, y_train.values.ravel())

MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(10, 10, 10), learning_rate='constant',
       learning_rate_init=0.001, max_iter=1850, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=None, shuffle=True, solver='adam', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)

The final step is to make predictions on our test data

predictions = mlp.predict(X_test)

Evaluating the Algorithm

Now it is time to evaluate performance of the model. Being a classification algorithm, the most commonly used metrics are a confusion matrix, precision, recall, and f1 score.

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))

[[16  2]
 [ 1  1]]

print(classification_report(y_test, predictions))

              precision    recall  f1-score   support

           0       0.94      0.89      0.91        18
           1       0.33      0.50      0.40         2

   micro avg       0.85      0.85      0.85        20
   macro avg       0.64      0.69      0.66        20
weighted avg       0.88      0.85      0.86        20

you can see from the confusion matrix


