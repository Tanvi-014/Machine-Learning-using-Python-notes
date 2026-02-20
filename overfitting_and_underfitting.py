from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot

x,y = make_classification(n_samples=9000,n_features=18,n_informative=4,n_redundant=12, random_state=4)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

"""train_scores, test_scores = list(), list()

values=[i for i in range(1,21)]

for i in values:
    model = DecisionTreeClassifier(max_depth = i)
    model.fit(x_train,y_train)
    train_yhat = model.predict(x_train)
    train_acc = accuracy_score(y_train,train_yhat)
    test_yhat = model.predict(x_test)
    test_acc = accuracy_score(y_test,test_yhat)
    train_scores.append(train_acc)
    test_scores.append(test_acc)
    print(f"{i}, train: {train_acc:.3f}, test: {test_acc:.3f}")

pyplot.plot(values, train_scores, '-o', label='Train')
pyplot.plot(values, test_scores, '-o', label='Test')
pyplot.legend()
pyplot.show()"""

## Overfitting: high accuracy for trained data but not test data
## Fixing overfitting prolems...

from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion' : ['entropy', 'gini'],
    'max_depth' : [2,4,6,10,20],
    'min_samples_split' : [5,10,20,50,100]
}

clf = GridSearchCV(DecisionTreeClassifier(),param_grid,cv=3,n_jobs=-1,scoring="accuracy")
clf.fit(x_train,y_train)

best_model = clf.best_estimator_
best_criterion = best_model.criterion
best_min_split = best_model.min_samples_split

train_scores, test_scores = [], []
values = range(1, 21)

for i in values:
    model = DecisionTreeClassifier(
        max_depth=i,
        criterion=best_criterion,
        min_samples_split=best_min_split
    )
    
    model.fit(x_train, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(x_train))
    test_acc = accuracy_score(y_test, model.predict(x_test))
    
    train_scores.append(train_acc)
    test_scores.append(test_acc)

pyplot.figure()
pyplot.plot(values, train_scores, '-o', label='Train')
pyplot.plot(values, test_scores, '-o', label='Test')
pyplot.axvline(x=best_model.max_depth, linestyle='--', label='Best Depth')
pyplot.legend()
pyplot.show()


