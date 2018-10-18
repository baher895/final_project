from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pickle
import math
import io

cluster_size = 300

tokenize = lambda doc: doc.lower().split(" ")

## get the doc in a list of vectors and return a list of number of closest word to each cluster center
def convert_to_cluster_distances(doc_vector):
    clsuter_statistics = [[]] * cluster_size
    distanceList = [[]] * cluster_size
    
    print('Convert Document with %d words!'%len(doc_vector))
    
    for word in doc_vector:
        index, distance = find_closest_cluster(word)
        distanceList[index].append(distance)
        
    for i in range(cluster_size):
        if len(distanceList[i]) != 0:
            clsuter_statistics[i].append(len(distanceList[i]))
            clsuter_statistics[i].append(min(distanceList[i]))
            clsuter_statistics[i].append(max(distanceList[i]))
            clsuter_statistics[i].append(sum(distanceList[i])/len(distanceList[i]))
        else:
            clsuter_statistics[i] = [0,0,0,0]

    return clsuter_statistics

## get a word and km to find the closest cluster center
def find_closest_cluster(word):
    min_distance = 0
    min_index = 0
    for index,center in enumerate(km.cluster_centers_):
        #distance = calculate_distance(word, center)
        dis = distance.cdist([word], [center], 'euclidean')
      
        if index == 0:
            min_distance = dis[0][0]
            min_index = 0
        elif dis[0][0] < min_distance:
            min_distance = dis[0][0];
            min_index = index;
    
    return min_index, min_distance


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

## reading the train and test sets
X_train = fetch_20newsgroups(subset='train' , remove=('headers', 'footers', 'quotes'))
X_test = fetch_20newsgroups(subset='test' , remove=('headers', 'footers', 'quotes'))

## define the train and test desired target
Y_train, Y_test = X_train.target, X_test.target

#convert each document to an array of words   
tokenized_documents_train = [tokenize(doc) for doc in X_train.data]
tokenized_documents_test = [tokenize(doc) for doc in X_test.data]

#unique words in whole rain Set
unique_words_in_train = set([word for doc in tokenized_documents_train for word in doc])
unique_words_in_test = set([word for doc in tokenized_documents_test for word in doc])

## load model
model_train = load_vectors("wiki-news-300d-1M-subword.vec", unique_words_in_train); # model[sample_word] has a 300 dim vector for word "sample_word"
print('Model for Training Data is loaded %d'%len(model_train))
model_test = load_vectors("wiki-news-300d-1M-subword.vec", unique_words_in_test); # model[sample_word] has a 300 dim vector for word "sample_word"
print('Model for Testing Data is loaded %d'%len(model_test))

        
print('X_KM is going to be produced')

unique_words_in_train = list(unique_words_in_train)
x_km = []
for word in unique_words_in_train:
    if word in model_train:
        x_km.append(model_train[word])

## train KMeans
km = MiniBatchKMeans(n_clusters = cluster_size).fit(x_km)
print('MiniBatchKMeans is All Set, ready to serve')

## prepare train docs
X_train_docs = []

for doc in X_train.data:
    doc_vector_train = []
    doc_tokenize = tokenize(doc)
    for word in doc_tokenize:
        if word in model_train:
            doc_vector_train.append(model_train[word])

    new_train_doc = convert_to_cluster_distances(doc_vector_train)
    X_train_docs.append(new_train_doc)
  
print('Train Doc is set')


## prepare test docs
X_test_docs = []

for doc in X_test.data:
    doc_vector_test = []
    doc_tokenize = tokenize(doc)
    for word in doc_tokenize:
        if word in test_model:
            doc_vector_test.append(model_test[word])

    new_test_doc = convert_to_cluster_distances(doc_vector_test)
    X_test_docs.append(new_test_doc)
  
print('Test Doc is set')


## final
clf = LogisticRegression(C=1, penalty='l2')
clf.fit(X_train_docs, Y_train)
pred = clf.predict(X_test_docs)
print(confusion_matrix(Y_test, pred))
print(accuracy_score(Y_test, pred))
