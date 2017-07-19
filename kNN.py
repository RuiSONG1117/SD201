
# coding: utf-8

# In[178]:

import os
import sys
import random

os.chdir("/Users/songsophie/Documents/SD/SD201 DataMining/TP2/data")

import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

def splitFileName(fileNames, labels):
    trainingFileNames = []
    trainingLabels = []
    testFileNames = []
    testLabels = []
    if(len(fileNames)<>len(labels)):
        print("Dimension wrong")
        return [],[],[],[]
    for i in range(0, len(fileNames)):
        k = random.random()
        if (k < 0.6667):
            trainingFileNames.append(fileNames[i])
            trainingLabels.append(labels[i])
        else:
            testFileNames.append(fileNames[i])
            testLabels.append(labels[i])
    return trainingFileNames, trainingLabels, testFileNames, testLabels


FileNames = ["apple1.txt","apple2.txt","apple3.txt","apple4.txt","apple5.txt",
             "apple6.txt","apple7.txt","apple8.txt","apple9.txt","apple10.txt",
             "apple11.txt","apple12.txt","apple13.txt","apple14.txt","apple15.txt"]
Labels = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1]

trainingFileNames, trainingLabels, testFileNames, testLabels = splitFileName(FileNames, Labels)
print trainingFileNames, trainingLabels, testFileNames, testLabels

# read the text of training documents
trainingFiles = []
# the i th training file
for i in range(0, len(trainingFileNames)):
    string = ""
    of = open(trainingFileNames[i], 'r')
    string = of.read().strip()
    #for line in of:
    #    string += line.strip()
    trainingFiles.append(string) 
#print trainingFiles[1]

#read the text of test documents
testFiles = []
for j in range(0, len(testFileNames)):
    string1 = ""
    of = open(testFileNames[j], 'r')
    string1 = of.read().strip()
    #for line1 in of:
     #   string1 += line1.strip()
    testFiles.append(string1)

count_vect = CountVectorizer(stop_words = 'english')

# convert training and test data to matrix
X_train_counts = count_vect.fit_transform(trainingFiles)
training = X_train_counts.toarray()
trainingCl = np.array(trainingLabels)

X_test_counts = count_vect.transform(testFiles)
test = X_test_counts.toarray()
testCl = np.array(testLabels)

print("Results using count_vect:")
for k in range(1,5):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(training,trainingCl)
    print("neighbors: %d, score: %f" %(k, neigh.score(test, testCl)))


# improve the classifier with tf-idf representation
tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
training_tfidf = X_train_tfidf.toarray()
trainingCl_tfidf = np.array(trainingLabels)

X_test_tfidf = tfidf_transformer.transform(X_test_counts)
test_tfidf = X_test_tfidf.toarray()
testCl_tfidf = np.array(testLabels)

print("Results using tf-idf:")
for k in range(1,5):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(training_tfidf,trainingCl_tfidf)
    print("neighbors: %d, score: %f" %(k, neigh.score(test_tfidf, testCl_tfidf)))


