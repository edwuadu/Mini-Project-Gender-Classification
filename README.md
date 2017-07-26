# Gender-Classification-
Python code using machine learning classifiers to classify gender by height, width, and shoe size.
First we do this!!
'''python
s = "
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# We create 4 classifiers
clf_tree = tree.DecisionTreeClassifier()
clf_knn =  KNeighborsClassifier(n_neighbors=3)
clf_randfor = RandomForestClassifier(n_estimators=11)
clf_SVC = SVC()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# ...and train them on our data
clf_tree = clf_tree.fit(X, Y)
clf_knn = clf_knn.fit(X, Y)
clf_randfor = clf_randfor.fit(X, Y)
clf_SVC = clf_SVC.fit(X, Y)

pred_tree = clf_tree.predict(X)
pred_knn = clf_knn.predict(X)
pred_randfor = clf_randfor.predict(X)
pred_SVC = clf_SVC.predict(X)

#compare results by testing accuracy
acc_tree = accuracy_score(Y, pred_tree)
acc_knn = accuracy_score(Y, pred_knn)
acc_randfor = accuracy_score(Y, pred_randfor)
acc_SVC = accuracy_score(Y, pred_SVC)

print ("Accuracy for Decision Tree is: ", acc_tree)
print ("Accuracy for K Nearest Neighbors is: ", acc_knn)
print ("Accuracy for Random Forest is: ", acc_randfor)
print ("Accuracy for SVC is: ", acc_SVC)

results = {'Decision Tree': 1.0, "K Nearest Neighbors" : 0.818181818182, "Random Forest": 1.0, "SVC":1.0 }
highest = max(results.values())
print("The best classifiers are: ",[k for k, v in results.items() if v == highest])"
print s
'''
