NumPy : Python library For fast numerical calculations
Pandas : Python library for working with structured data
Scikit-learn: (aka sklearn) is a Python library for machine learning.
Seaborn : Python library for statistical data visualization built on top of matplotlib
Statsmodels : Python library for statistical modeling and hypothesis testing (regression, p-values, inference)


## Basic Data Handling Functions

1. df(dataframe) = pd.read_csv('name of file',sep=',')
> *[reads a csv file (in table format) file]*
> *[sep = separation of data]*

2. df.head 
> *[ First 5 rows of table]*

3. df.info 
> *[basic info about this table]*

4. np.where(condition, value_if_true, value_if_false)

5. df.drop(['columns/rows to drop/remove'], axis=1)
> *[axis=0(rows) axis=1(columns)]*

6. df.isna().sum()
> *isna->This turns your table into True / False values.*
> *“Count how many missing values are in each column.”*
> *Because Pandas’ default is: operate column-wise.*

7. preprocessing
>preparing data before ML
>Before a model learns, we:
- clean data
- fix missing values
- scale numbers

8. MinMaxScaler: data preprocessing technique that rescales numerical features to a fixed range, typically [0, 1]
9. StandardScaler: Center numbers around 0 and spread them evenly