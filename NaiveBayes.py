import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

def k_fold_NB_accuracy(data, classes, k):
    if(len(data)!=len(classes)):
        return -1;
    else:
        n = len(data)/k
        error = 0;
        for i in range(0, k):
            X_test = data[0:n]
            X_test_Cl = classes[0:n]
            X_train = data[n:]
            X_train_Cl = classes[n:]
            
            clf = MultinomialNB()
            clf.fit(X_train, X_train_Cl)
            result = clf.predict(X_test)
            
            for j in range(0, len(X_test_Cl)):
                if(X_test_Cl[j] != result[j]):
                    error += 1
        
            data = np.append(X_train, X_test, axis = 0)
        classes = np.append(X_train_Cl, X_test_Cl, axis = 0)
            return  1 - float(error)/(n*k)


files = ['apple_the_company.txt', 'apple_the_fruit.txt', 'banana.txt', 'microsoft.txt']
dataFiles = []
dataClasses = []

for i in range(0, len(files)):
    f = open(files[i], 'r')
    for line in f:
        dataFiles.append(line)
        dataClasses.append(i)

count_vect = CountVectorizer(stop_words='english')
vectors = count_vect.fit_transform(dataFiles)
X = vectors.toarray()

np.random.seed(30)
X = np.random.permutation(X)
np.random.seed(30)
dataClasses = np.random.permutation(dataClasses)



print k_fold_NB_accuracy(X, dataClasses, 2)
print k_fold_NB_accuracy(X, dataClasses, 10)
print k_fold_NB_accuracy(X, dataClasses, 100)