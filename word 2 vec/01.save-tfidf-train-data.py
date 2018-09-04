import pickle
import tfidf
from sklearn.datasets import fetch_20newsgroups

X_train = fetch_20newsgroups(subset='train' , remove=('headers', 'footers', 'quotes'))
tfidf.run(X_train.data, "train"); # pass all documents in test set to tfidf
