import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('position_salaries.csv')

df.info()

X=df.iloc[:,1:2].values
y=df.iloc[:,2].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.6, random_state=0)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

LinearRegression()

def viz_linear():
    plt.scatter(X,y,color='red')
    plt.plot(X, lin_reg.predict(X),color='blue')
    plt.title('Linear regression model')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    return

viz_linear()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

LinearRegression()

def viz_polynomial():
    plt.scatter(X,y,color='red')
    plt.plot(X,pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title('Polynomial regression')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    return

viz_polynomial()

print(lin_reg.predict([[5.5]]))

print(pol_reg.predict(poly_reg.fit_transform([[5.5]])))