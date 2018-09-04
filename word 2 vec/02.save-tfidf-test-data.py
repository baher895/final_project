import pickle
import tfidf
from sklearn.datasets import fetch_20newsgroups

X_test = fetch_20newsgroups(subset='test' , remove=('headers', 'footers', 'quotes'))
tfidf.run(X_test.data, "test"); # pass all documents in training set to tfidf method
