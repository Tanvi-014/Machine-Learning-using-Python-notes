import numpy as np
import pandas as pd 
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot
import seaborn as sns
import statsmodels.api as sm

from sklearn.datasets import fetch_california_housing
housing=fetch_california_housing()

#Exploratory Data Analysis (EDA)
#Answers the question - What does this housing dataset look like?

housing_df = pd.DataFrame(
    data=housing.data,
    columns=housing.feature_names)

housing_df['HousePrice'] = housing.target 

housing_df.head()
housing_df.describe()

housing_df.isna().sum()

sns.pairplot(housing_df, height=3)
#This creates MANY small scatter plots.
#It shows: Each feature vs every other feature, to visually see relationships.

pyplot.suptitle('Pair Plot of Features against HousePrice',y=1.02)


housing_df.plot(
    kind="scatter",
    x="Longitude",
    y="Latitude",
    alpha=0.4,
    s=housing_df['Population']/100,
    label="Population",
    c="HousePrice",
    cmap="jet",
    colorbar=True,
    figsize=(10,8)
)
#This reveals geographic trends, e.g - Are expensive houses clustered in specific locations?
#Blue → cheaper houses
#Green → mid-priced
#Yellow/Red → expensive houses

corr = housing_df.corr()
pyplot.figure(figsize=(10,8))

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidth=0.5)
#The heatmap shows:
#Red → strong positive relationship
#Blue → strong negative relationship
#Near white → weak relationship

corr["HousePrice"].abs().sort_values(ascending=False)
#Doesn't ca

pyplot.show()