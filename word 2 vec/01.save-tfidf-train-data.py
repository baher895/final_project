import pickle
import tfidf1
from sklearn.datasets import fetch_20newsgroups

X_train = fetch_20newsgroups(subset='train' , remove=('headers', 'footers', 'quotes'))
tfidf1.run(X_train.data, "train1"); # pass all documents in test set to tfidf
