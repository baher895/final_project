import math
import pickle

tokenize = lambda doc: doc.lower().split(" ")

def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return math.log(1 + count)


def inverse_document_frequencies(tokenized_documents):
    idf_values = {}

    #unnique words in whole documents
    all_tokens_set = set([word for doc in tokenized_documents for word in doc])
    print('unique word in all documents %d'%len(all_tokens_set))
    con = 1
        
    for tkn in all_tokens_set:
        print(con)
        con += 1
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = math.log(1 + len(tokenized_documents)/(sum(contains_token)))
        
    return idf_values

def run(documents, dataSet):
    print('tf-idf starts')
    
    #convert each document to an array of words   
    tokenized_documents = [tokenize(d) for d in documents]

    #calculate idf for each word in all documents
    idf = inverse_document_frequencies(tokenized_documents)
    
    tfidf_documents = []
    
    count = 0
    for document in tokenized_documents:
        doc_tfidf = {}
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf[term] = tf * idf[term]
        
       
        fileName = dataSet + "/" + str(count) + ".pkl"
        output = open(fileName, 'wb')
        pickle.dump(doc_tfidf, output)
        output.close()
        count += 1

    print('Finally, its Finished!')
    
    return
