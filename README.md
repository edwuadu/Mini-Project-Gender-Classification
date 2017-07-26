# Gender-Classification-
This is a script written in python that can classify anyone as male or female given their body measurements, specifically being height, width, and shoe size. For this mini project, we wish to compare different machine learning classification models to test accuracy.

First we import our submodules from scikit learn, which will allow us to use our machine learning classifier models.
```python
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
```  
We can see from above that we've imported tree, KNeighborsClassifier, RandomForestClassifier, and SVC. Respectively, the machine learning models we will be using will be Decision Tree, K-Nearest Neighbors, Random Forest, and Support Vector Classification.  
```python
# We create 4 classifiers
clf_tree = tree.DecisionTreeClassifier()
clf_knn =  KNeighborsClassifier(n_neighbors=3)
clf_randfor = RandomForestClassifier(n_estimators=11)
clf_SVC = SVC()
```  

We will create a small dataset of 11 participants, storing their body measurements in a list we name as X. The variably Y will hold the gender labels for each of the entries in X.  
```python
# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']
```  
We use the fit function to train the classifiers we previously initialized on our training set.
```python
# ...and train them on our data
clf_tree = clf_tree.fit(X, Y)
clf_knn = clf_knn.fit(X, Y)
clf_randfor = clf_randfor.fit(X, Y)
clf_SVC = clf_SVC.fit(X, Y)
```  
We then predict the gender of the same dataset X, using our machine learning models.
```python
pred_tree = clf_tree.predict(X)
pred_knn = clf_knn.predict(X)
pred_randfor = clf_randfor.predict(X)
pred_SVC = clf_SVC.predict(X)
```  
Now we test for accuracy comparing the gender results given by the classification methods to the actual gender labels Y.
```python
#compare results by testing accuracy
acc_tree = accuracy_score(Y, pred_tree)
acc_knn = accuracy_score(Y, pred_knn)
acc_randfor = accuracy_score(Y, pred_randfor)
acc_SVC = accuracy_score(Y, pred_SVC)
```
The results are as follows, and we can see which machine learning methods were the most accurate in classifiying gender.
![](~/Desktop/Screen Shot 2017-07-26 at 2.56.40 PM.png)
