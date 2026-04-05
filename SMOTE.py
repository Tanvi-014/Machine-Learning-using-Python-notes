import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('cuisines.csv')

df.head()
df.info()

df.cuisine.value_counts().plot.barh()

thai_df = df[(df.cuisine == 'thai')]
japanese_df = df[(df.cuisine == 'japanese')]
chinese_df = df[(df.cuisine == 'chinese')]
indian_df = df[(df.cuisine == 'indian')]
korean_df = df[(df.cuisine == 'korean')]

print(f'thai df: {thai_df.shape}')
print(f'japanese df: {japanese_df.shape}')
print(f'chinese df: {chinese_df.shape}')
print(f'indian df: {indian_df.shape}')
print(f'korean df: {korean_df.shape}')

def create_ingredient_df(df):
    ingredient_df = df.T.drop(['cuisine', 'Unnamed: 0']).sum(axis=1).to_frame('value')
    ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
    ingredient_df = ingredient_df.sort_values(by='value', ascending = False, inplace = False)
    return ingredient_df

thai_ingredient_df = create_ingredient_df(thai_df)
thai_ingredient_df.head(10).plot.barh()
plt.show()

japanese_ingredient_df = create_ingredient_df(japanese_df)
japanese_ingredient_df.head(10).plot.barh()
plt.show()

chinese_ingredient_df = create_ingredient_df(chinese_df)
chinese_ingredient_df.head(10).plot.barh()
plt.show()

indian_ingredient_df = create_ingredient_df(indian_df)
indian_ingredient_df.head(10).plot.barh()
plt.show()

korean_ingredient_df = create_ingredient_df(korean_df)
korean_ingredient_df.head(10).plot.barh()
plt.show()

feature_df = df.drop(['cuisine', 'Unnamed: 0', 'rice', 'garlic', 'ginger'], axis=1)
## we are trying to predict cuisine from ingredients so remove the common ones 
## we also remove the cuisine column so the model can't peak at the answer
labels_df = df.cuisine
feature_df.head()

##smote is used to balance an unbalanced data set
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE 
oversample = SMOTE()
transform_feature_df, transformed_label_df = oversample.fit_resample(feature_df,labels_df)

print(f'new label count:  {transformed_label_df.value_counts()}')
print(f'old label count:  {labels_df.value_counts()}')

X_train, X_test, y_train, y_test = train_test_split(transform_feature_df, transformed_label_df, test_size=0.3)

lr=LogisticRegression(solver='lbfgs', max_iter=1000)
model=lr.fit(X_train, np.ravel(y_train))

accuracy = model.score(X_test, y_test)
print("Accuracy is {}".format(accuracy))

print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
print(f'cuisine: {y_test.iloc[50]}')

test = X_test.iloc[50].values.reshape(-1,1).T
proba = model.predict_proba(test)
classes = model.classes_
resultdf = pd.DataFrame(data=proba, columns=classes)

toppred = resultdf.T.sort_values(by=[0], ascending=[False])
toppred.head()

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
