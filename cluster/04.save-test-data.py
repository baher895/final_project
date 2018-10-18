from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import fetch_20newsgroups
import pickle
import math

cluster_size = 300
tokenize = lambda doc: doc.lower().split(" ")

## get the doc in a list of vectors and return a list of number of closest word to each cluster center
def convert_to_cluster_distances(doc_vector):
    clsuter_statistics = [[]] * cluster_size
    distanceList = [[]] * cluster_size

    print('Convert %d'%len(doc_vector))
    
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
        distance = calculate_distance(word, center)
      
        if index == 0:
            min_distance = distance
            min_index = 0
        elif distance < min_distance:
            min_distance = distance;
            min_index = index;
    
    return min_index, min_distance

## calculate distance between two vector
def calculate_distance(p1,p2):
    dis = 0
    for index in range(cluster_size):
        dis += ( p1[index] - p2[index] ) ** 2

    return math.sqrt(dis)

X_train = fetch_20newsgroups(subset='train' , remove=('headers', 'footers', 'quotes'))

## load x_km_training
fileName = "X_KM_Train.pkl"
X_KM_file = open(fileName, 'rb')
x_km_train = pickle.load(X_KM_file)
X_KM_file.close()
print('X_KM is loaded!!!!')

## train KMeans
km = MiniBatchKMeans(n_clusters = cluster_size).fit(x_km_train)
print('MiniBatchKMeans is All Set, ready to serve')

## load test Model
fileName = "test-model.pkl"
test_model_file = open(fileName, 'rb')
test_model = pickle.load(test_model_file)
test_model_file.close()
print('Test Model loaded!!')

## prepare test docs
X_test_docs = []

for doc in X_test.data:
    doc_vector_test = []
    doc_tokenize = tokenize(doc)
    for word in doc_tokenize:
        if word in test_model:
        doc_vector_test.append(test_model[word])
          
    new_test_doc = convert_to_cluster_distances(doc_vector_test)
    X_test_docs.append(new_test_doc)
  
print('Test Doc is set')

fileName = "test-data.pkl"
output = open(fileName, 'wb')
pickle.dump(X_test_docs, output)
output.close()

print('Test Doc is Saved')
