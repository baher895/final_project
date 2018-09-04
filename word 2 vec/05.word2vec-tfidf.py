from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pickle

	
X_train = fetch_20newsgroups(subset='train' , remove=('headers', 'footers', 'quotes'))
X_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

Y_train, Y_test = X_train.target, X_test.target

print("%d documents  (training set)" %  len(X_train.data))
print("%d documents  (test set)" % len(X_test.data))
print("******************************")


fileName = "train-vec.pkl"
train_file = open(fileName, 'rb')
X_train = pickle.load(train_file)
train_file.close()
print('Train Set is Loaded')


fileName = "test-vec.pkl"
test_file = open(fileName, 'rb')
X_test = pickle.load(test_file)
test_file.close()
print('Test Set is Loaded')


print(X_train.shape)

clf = LogisticRegression(C=100, penalty='l1')
clf.fit(X_train, Y_train)
pred = clf.predict(X_test)
print(confusion_matrix(Y_test, pred))
print(accuracy_score(Y_test, pred))
