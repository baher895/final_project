from sklearn.datasets import fetch_20newsgroups
import pickle
import io

tokenize = lambda doc: doc.lower().split(" ")

## load per-tested data
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

X_test = fetch_20newsgroups(subset='test' , remove=('headers', 'footers', 'quotes'))

#convert each document to an array of words   
tokenized_documents = [tokenize(doc) for doc in X_test.data]

#unique words in whole rain Set
unique_words_in_test = set([word for doc in tokenized_documents for word in doc])

## load model
model = load_vectors("wiki-news-300d-1M-subword.vec", unique_words_in_test); # model[sample_word] has a 300 dim vector for word "sample_word"
print('Model for testing Data is loaded %d'%len(model))

fileName = "test-model.pkl"
output = open(fileName, 'wb')
pickle.dump(model, output)
output.close()
        
print('test Model is Saved')
