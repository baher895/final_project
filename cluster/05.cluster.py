from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pickle

        
## reading the train and test sets
X_train = fetch_20newsgroups(subset='train' , remove=('headers', 'footers', 'quotes'))
X_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

## define the train and test desired target
Y_train, Y_test = X_train.target, X_test.target


#### ---------------
#### load train doc
#### ---------------
fileName = "trained-data.pkl"
X_train_file = open(fileName, 'rb')
X_train_docs = pickle.load(X_train_file)
X_train_file.close()
print('Train Data is  loaded!!')

#### ---------------
#### load test doc 
#### ---------------
fileName = "test-data.pkl"
X_test_file = open(fileName, 'rb')
X_test_docs = pickle.load(X_test_file)
X_test_file.close()
print('Train Data is  loaded!!')

## final
clf = LogisticRegression(C=1, penalty='l2')
clf.fit(X_train_docs, Y_train)
pred = clf.predict(X_test_docs)
print(confusion_matrix(Y_test, pred))
print(accuracy_score(Y_test, pred))
