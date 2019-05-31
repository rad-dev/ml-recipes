from sklearn import tree

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]

classifier = tree.DecisionTreeClassifier()
classifier.fit(features, labels)

# should print [1, 0] for orange and apple respectively
print(classifier.predict([[150, 0], [120, 1]]))

