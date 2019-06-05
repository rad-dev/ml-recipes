from sklearn import datasets, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

tree_clf = tree.DecisionTreeClassifier()
tree_clf.fit(x_train, y_train)

print('******DECISION TREE CLASSIFIER******')
predictions = tree_clf.predict(x_test)
print(predictions)
print(y_test)
print(accuracy_score(y_test, predictions))

print('******K NEAREST NEIGHBORS CLASSIFIER******')
n_clf = KNeighborsClassifier()
n_clf.fit(x_train, y_train)
n_predictions = n_clf.predict(x_test)
print(n_predictions)
print(y_test)
print(accuracy_score(y_test, n_predictions))