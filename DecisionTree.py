import sklearn as sk
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import numpy as np
import pydotplus
from scipy import stats

training_data = np.loadtxt('training_data.txt', delimiter=' ')
training_class = np.loadtxt('training_class.txt', delimiter=' ')
test_data = np.loadtxt('test_data.txt', delimiter=' ')
test_class = np.loadtxt('test_class.txt', delimiter=' ')

clf = DecisionTreeClassifier(criterion = 'gini', random_state=np.random.RandomState(130))
clf = clf.fit(training_data, training_class)

dot_data = export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("DecisionTree.pdf")

result = clf.predict(test_data)
a = []
for i in range(0, len(result)):
    if(result[i] == test_class[i]):
        a.append(1)
    else:
        a.append(0)

print "Confidence interval:"
print stats.norm.interval(0.95, loc=np.mean(a), scale=stats.sem(a))

clf2 = DecisionTreeClassifier(criterion = 'gini', 
                              random_state=0, 
                              min_impurity_split = 0.5, 
                              presort = True )
clf2 = clf2.fit(training_data, training_class)

dot_data2 = export_graphviz(clf2, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data2)
graph.write_pdf("DecisionTree2.pdf")

result = clf2.predict(test_data)
a = []
for i in range(0, len(result)):
    if(result[i] == test_class[i]):
        a.append(1)
    else:
        a.append(0)

print "Imporved confidence interval:"
print stats.norm.interval(0.95, loc=np.mean(a), scale=stats.sem(a))

