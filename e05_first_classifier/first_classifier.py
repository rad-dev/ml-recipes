import random
from scipy.spatial import distance


def euclidean_dist(a, b):
    return distance.euclidean(a, b)


class MyKNN():
    def fit(self, x, y):
        self.X_train = x
        self.y_train = y 
    
    def predict(self, x_test):
        predictions = []

        for row in x_test:
            label = self.closest(row)
            predictions.append(label)

        return predictions

    def closest(self, row):
        best_distance = euclidean_dist(row, self.X_train[0])
        best_index = 0

        for i in range(1, len(x_train)):
            dist = euclidean_dist(row, x_train[i])
            
            if dist < best_distance:
                best_distance = dist
                best_index = i

        return self.y_train[best_index]

            

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
 
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = MyKNN()
clf.fit(x_train, y_train)

predictions = clf.predict(x_test)
print(predictions)
print(y_test)
print(accuracy_score(y_test, predictions))
