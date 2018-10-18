from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import io

tokenize = lambda doc: doc.lower().split(" ")

def avg_word_vectors(wordlist, count, mode):

    tokenized_wordlist = tokenize(wordlist)

    sumvec = np.zeros(shape=(1,300));
    wordcnt = 0;
    weight = 0;
        
    for word in tokenized_wordlist:
        if mode == 'train' and word in model_train and word in vectorizer.vocabulary_:
            weight = transformed_train[count, vectorizer.vocabulary_[word]]
            sumvec += np.array(model_train[word])* weight
            wordcnt += weight
        elif mode == 'test' and word in model_test and word in vectorizer.vocabulary_:
            weight = transformed_test[count, vectorizer.vocabulary_[word]]
            sumvec += np.array(model_test[word])* weight
            wordcnt += weight
            
            
    if wordcnt ==0:
        return sumvec
    else:
        return sumvec / wordcnt

## load per-trained data
def load_vectors(fname, word_dictionary):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        if tokens[0] in word_dictionary:
            data[tokens[0]] = list(map(float, tokens[1:]))
            
 
    return data

# X
X_train = fetch_20newsgroups(subset='train' , remove=('headers', 'footers', 'quotes'))
X_test = fetch_20newsgroups(subset='test' , remove=('headers', 'footers', 'quotes'))

# Y, correct target
Y_train, Y_test = X_train.target, X_test.target

# tfidf
vectorizer = TfidfVectorizer( sublinear_tf= False, analyzer='word', ngram_range=(1,2),min_df=1, max_df=0.8, stop_words='english')

# train
transformed_train = vectorizer.fit_transform(X_train.data)

# test
transformed_test  = vectorizer.transform(X_test.data)

#convert each document to an array of words   
tokenized_documents_train = [tokenize(doc) for doc in X_train.data]
tokenized_documents_test = [tokenize(doc) for doc in X_test.data]

#unique words in whole rain Set
unique_words_in_train = set([word for doc in tokenized_documents_train for word in doc])
unique_words_in_test = set([word for doc in tokenized_documents_test for word in doc])

# load model
model_train = load_vectors("wiki-news-300d-1M-subword.vec", unique_words_in_train); # model[sample_word] has a 300 dim vector for word "sample_word"
print('Model for Training Data is loaded %d'%len(model_train))
model_test = load_vectors("wiki-news-300d-1M-subword.vec", unique_words_in_test); # model[sample_word] has a 300 dim vector for word "sample_word"
print('Model for Test Data is loaded %d'%len(model_test))

X_train = np.concatenate([avg_word_vectors(doc,count, 'train') for count, doc in enumerate(X_train.data)]) # baraye har document anjam midim
X_test = np.concatenate([avg_word_vectors(doc,count, 'test') for count, doc in enumerate(X_test.data)]) # baraye har document anjam midim

# accuracy
print('Check Accuracy ...')
clf = LogisticRegression(C=0.8, penalty='l2')
clf.fit(X_train, Y_train)
pred = clf.predict(X_test)
print(confusion_matrix(Y_test, pred))
print(accuracy_score(Y_test, pred))
