import pickle
import tfidf1
from sklearn.datasets import fetch_20newsgroups

X_test = fetch_20newsgroups(subset='test' , remove=('headers', 'footers', 'quotes'))
tfidf1.run(X_test.data, "test1"); # pass all documents in training set to tfidf method
