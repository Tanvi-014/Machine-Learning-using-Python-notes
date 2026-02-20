from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot

x,y = make_classification(n_samples=9000,n_features=18,n_informative=4,n_redundant=12, random_state=4)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)