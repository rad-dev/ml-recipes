from sklearn import tree


# first number => weight, second number is texture (1 = smooth, 0 = bumpy)
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# 0 = apple, 1 = orange
labels = [0, 0, 1, 1]

classifier = tree.DecisionTreeClassifier()
classifier.fit(features, labels)

# should print [1, 0] for orange and apple respectively
print(classifier.predict([[150, 0], [120, 1]]))

