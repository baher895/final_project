from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pickle
import io

tokenize = lambda doc: doc.lower().split(" ")

def avg_word_vectors(wordModel, wordlist, size, count, dataSet):

    tokenized_wordlist = tokenize(wordlist)
    
    fileName = dataSet + "/" + str(count) + ".pkl"
    tfidf_file = open(fileName, 'rb')
    tfidf = pickle.load(tfidf_file)
    tfidf_file.close()

    sumvec = np.zeros(shape=(1,size));
    wordcnt = 0;
    weight = 0;
    
    for word in tokenized_wordlist:
        if word in model:
            weight = tfidf[word]
            sumvec += np.array(model[word])* weight
            wordcnt += weight

    print(wordcnt)
    if wordcnt ==0:
        return sumvec
    else:
        return sumvec / wordcnt

## load per-tested data
def load_vectors(fname, word_dictionary):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        if tokens[0] in word_dictionary:
            data[tokens[0]] = list(map(float, tokens[1:]))
            
 
    return data

X_test = fetch_20newsgroups(subset='test' , remove=('headers', 'footers', 'quotes'))

#convert each document to and array of words   
tokenized_documents = [tokenize(doc) for doc in X_test.data]

#unnique words in whole rain Set
unique_words_in_test = set([word for doc in tokenized_documents for word in doc])

model = load_vectors("wiki-news-300d-1M-subword.vec", unique_words_in_test); # model[sample_word] has a 300 dim vector for word "sample_word"
print('Model for testing Data is loaded %d'%len(model))

X_test = np.concatenate([avg_word_vectors(model, doc, 300,count, "test") for count, doc in enumerate(X_test.data)]) # baraye har document anjam midim

fileName = "test-vec.pkl"
output = open(fileName, 'wb')
pickle.dump(X_test, output)
output.close()
        
print('test Set is Saved')
