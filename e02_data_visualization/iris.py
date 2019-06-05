import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

# each type contains 50 elements so I'm getting the first items for each species of iris
test_index = [0, 50, 100]

x_train = np.delete(iris.data, test_index, axis=0)
y_train = np.delete(iris.target, test_index)

x_test = iris.data[test_index]
y_test = iris.target[test_index]

classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)

# these two prints should be equal values
print(y_test)
print(classifier.predict(x_test))

import graphviz 
dot_data = tree.export_graphviz(classifier, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 

dot_data = tree.export_graphviz(classifier, out_file=None, 
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph
