#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 21:15:58 2020

@author: aditya
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_gd = pd.read_csv('eda_data.csv')

df_model = df_gd[['Rating','Size','Type of ownership', 'Industry', 'Sector', 'Revenue', 
       'hourly', 'emp prov', 'avg_sal','state', 'hq_loc', 'age_comp', 'python_ys', 
       'excel_ys', 'tableau_ys','job_simplified', 'seniority', 'desc_len', 
       'num_competitors']]

# create dummy variables
df_dum = pd.get_dummies(df_model)

# train test split
from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_sal', axis = 1)
y = df_dum.avg_sal.values

x_train,x_test, y_train, y_test = train_test_split(X,y, test_size = .2, random_state = 0) 

# Linear regression stats model
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X)

model.fit().summary()

# Linear regression sklearn
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lin_reg = LinearRegression()
lin_reg.fit(x_train,y_train)

np.mean(cross_val_score(lin_reg, x_train,y_train, 
                        scoring='neg_mean_absolute_error', cv = 3))

# Lass model
lass = Lasso(alpha=0.07)
lass.fit(x_train,y_train)

np.mean(cross_val_score(lass, x_train,y_train, 
                        scoring='neg_mean_absolute_error', cv = 3))

# alpha = []
# error = []

# for i in range(1,100):
#     alpha.append(i/100)
#     lass_l = Lasso(alpha = i/100)
#     error.append(np.mean(cross_val_score(lass_l, x_train,y_train, 
#                         scoring='neg_mean_absolute_error', cv = 3)))

# plt.plot(alpha,error)   

# df_lass_error = pd.DataFrame(tuple(zip(alpha,error)), columns=['alpha','error'])

# df_lass_error[df_lass_error.error == max(df_lass_error.error)]

# alpha of 0.07 makes best error value 

# random forest
from sklearn.ensemble import RandomForestRegressor

rand = RandomForestRegressor()

np.mean(cross_val_score(rand, x_train,y_train, 
                        scoring='neg_mean_absolute_error', cv = 3))

# grid search for para tuning 
from sklearn.model_selection import GridSearchCV

params = {'n_estimators': range(10,100,10),
          'criterion': ('mse','mae'),
          'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(rand, params, scoring='neg_mean_absolute_error', cv=3)

gs.fit(x_train,y_train)

gs.best_score_

# test ensemble
from sklearn.metrics import mean_absolute_error

# Linear Regression
tpred_reg = lin_reg.predict(x_test)
mean_absolute_error(y_test, tpred_reg)

# Lasso Regression
tpred_lass = lass.predict(x_test)
mean_absolute_error(y_test, tpred_lass)

# Random forest
tpred_rf = gs.best_estimator_.predict(x_test)
mean_absolute_error(y_test, tpred_rf)

