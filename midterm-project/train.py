#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

# parameters
C = 1.0  # default
output_file = 'model.bin'

# data preparation
print('Data Preparation...')
print()

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

del df['id']
df['ever_married'] = (df['ever_married'] == 'Yes').astype(int)
df.columns = df.columns.str.lower()
for column in list(df.dtypes[df.dtypes == 'object'].index):
    df[column] = df[column].str.replace(' ', '_').str.replace('-', '_').str.lower()

numerical = list(df.columns[df.dtypes != object])
numerical.remove('stroke')
categorical = list(df.columns[df.dtypes == object])

df = df.drop(df[df['gender'] == 'other'].index)

df_full_train, df_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train['stroke']
y_test = df_test['stroke']

del df_full_train['stroke']
del df_test['stroke']

df_full_train['bmi'].fillna(df_full_train['bmi'].mean(), inplace=True)
df_test['bmi'].fillna(df_test['bmi'].mean(), inplace=True)

# training the final model
print('Training the model...')
print()

dv = DictVectorizer(sparse=False)

full_train_dict = df_full_train[categorical + numerical].to_dict(orient='records')
X_full_train = dv.fit_transform(full_train_dict)
model = LogisticRegression(C=C, max_iter=1000)
model.fit(X_full_train, y_full_train)

test_dict = df_test[categorical + numerical].to_dict(orient='records')
X_test = dv.transform(test_dict)
y_pred = model.predict_proba(X_test)[:, 1]
score = round(roc_auc_score(y_test, y_pred), 3)

print(f'Final Logistic Regression model score on a Test dataset is {score}')
print()

# saving the model
print('The model is saved to "model.bin"')
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)