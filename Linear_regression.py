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
#Doesn't care about positive or negative only checks strength of relationship

medinc = housing_df['MedInc']
##because medinc has greatest strength
houseprice = housing_df['HousePrice']

pyplot.figure(figsize=(8,6))
pyplot.scatter(medinc,houseprice,alpha=0.5,color='blue')
pyplot.title('Scatter plot of medinc vs houseprice')
pyplot.xlabel('medinc')
pyplot.ylabel('houseprice')
pyplot.grid(True)
pyplot.show()

#Variance inflation factor

from statsmodels.stats.outliers_influence import variance_inflation_factor

housing_df_vif = housing_df.drop('HousePrice',axis=1)
#removing the target variable because we calculate vif based on independent variables
#axis=1 : column 

housing_df_vif = housing_df_vif.apply(pd.to_numeric,errors='coerce')

vif_data = pd.DataFrame() #make it a table
vif_data["Feature"] = housing_df_vif.columns
vif_data["VIF"] = [variance_inflation_factor(housing_df_vif.values,i) for i in range (len(housing_df_vif.columns))]
#take feature number i and check how much the other features can explain it.
print(vif_data)

#VIF is not good for regression so we attempt to remove latitude and longitude columns as a result of the above vif test

housing_df = housing_df.drop(['Latitude', 'Longitude'], axis=1)

cols_to_convert = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
housing_df[cols_to_convert] = housing_df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
housing_df['HousePrice'] = pd.to_numeric(housing_df['HousePrice'], errors='coerce')
#Regression math needs numbers only.

x = sm.add_constant(housing_df.drop('HousePrice', axis=1))
y = housing_df['HousePrice']

model = sm.OLS(y,x).fit()
#Ordinary Least Squares: Use X to predict y.
#least error path (bw actual and prediction)

residuals = model.resid
#Residual = Actual value − Predicted value

pyplot.figure(figsize=(10,5))
pyplot.subplot(1, 2, 1) #1st
sns.histplot(residuals, kde=True)
#it groups nearby numbers together and shows how many fall in each group.
pyplot.title('Residuals distribution')

pyplot.subplot(1, 2, 2) #2nd
pyplot.scatter(model.predict(x), residuals)
pyplot.axhline(y=0, color='red', linestyle='--')
pyplot.title('Residuals vs Predicted')
pyplot.ylabel('Residuals')
pyplot.show()

print("Model coefficients: \n", model.params)

#Positive → model predicted too low
#Negative → model predicted too high
#Close to 0 → good prediction

from sklearn.metrics import mean_squared_error, r2_score

predictions = model.predict(x)

print("R square score:", r2_score(y, predictions))
print("R square score:", r2_score(y, predictions))

import matplotlib.pyplot as plt

def plot_boxplots(df):
    fig,axes = plt.subplots(1, len(df.columns), figsize=(20,5))
    
    for i, column_name in enumerate(df.columns):
        df.boxplot(column=column_name, ax=axes[i])
        axes[i].set_title(column_name)
    
    plt.show

plot_boxplots(housing_df)

#We see outliers in some features and need to correct them for accuracy of our model

def remove_outliers(df):
    df_no_outliers = df.copy()

    for column_name in df.columns:
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_no_outliers[column_name] = df_no_outliers[column_name].apply(lambda x: lower_bound if x < lower_bound
                                                                        else upper_bound if x > upper_bound else x)
        return df_no_outliers
    
    housing_df_new = remove_outliers(housing_df)

    plot_boxplots(housing_df_new)

    