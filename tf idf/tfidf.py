from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



data_train = fetch_20newsgroups(subset='train' , remove=('headers', 'footers', 'quotes'))
data_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

target_train, target_test = data_train.target, data_test.target

print("%d documents  (training set)" %  len(data_train.data))
print("%d documents  (test set)" % len(data_test.data))
print("******************************")

vectorizer = TfidfVectorizer( sublinear_tf=True, analyzer='word', ngram_range=(1,2),min_df=1, max_df=0.8, stop_words='english')

transformed_train = vectorizer.fit_transform(data_train.data) 
feature_names = vectorizer.get_feature_names() 

transformed_test  = vectorizer.transform(data_test.data)

print("Vocabulary Size: %d"% len(vectorizer.vocabulary_))
        
clf = LogisticRegression(C=20, penalty='l2')
clf.fit(transformed_train, target_train)
pred = clf.predict(transformed_test)
print(confusion_matrix(target_test, pred))
print(accuracy_score(target_test, pred))
