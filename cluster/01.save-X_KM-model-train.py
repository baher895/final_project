from sklearn.datasets import fetch_20newsgroups
import pickle
import io

tokenize = lambda doc: doc.lower().split(" ")

## load per-trained data
def load_vectors(fname, word_dictionary):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        if tokens[0] in word_dictionary:
            data[tokens[0]] = list(map(float, tokens[1:]))
            print(tokens[0])
            
 
    return data

X_train = fetch_20newsgroups(subset='train' , remove=('headers', 'footers', 'quotes'))

#convert each document to an array of words   
tokenized_documents = [tokenize(doc) for doc in X_train.data]

#unique words in whole rain Set
unique_words_in_train = set([word for doc in tokenized_documents for word in doc])

## load model
model = load_vectors("wiki-news-300d-1M-subword.vec", unique_words_in_train); # model[sample_word] has a 300 dim vector for word "sample_word"
print('Model for Training Data is loaded %d'%len(model))

fileName = "train-model.pkl"
output = open(fileName, 'wb')
pickle.dump(model, output)
output.close()
        
print('Train Model is Saved')
print('X_KM is going to be produced')

unique_words_in_train = list(unique_words_in_train)
x_km_train = []
for word in unique_words_in_train:
    if word in model:
        x_km_train.append(model[word])


fileName = "X_KM_Train.pkl"
output = open(fileName, 'wb')
pickle.dump(x_km_train, output)
output.close()
        
print('Train KM is Saved')
