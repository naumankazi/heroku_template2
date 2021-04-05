import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle

dataset = pd.read_csv('Engineering_graduate_salary.csv')
print(dataset.head())
print(dataset.info())


X = dataset.iloc[:, 0:-1].values
print(X)

Y = dataset.iloc[:, -1].values
print(Y)

regressor = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=63)
regressor.fit(X, Y)


pickle.dump(regressor, open('model.pkl', 'wb'))
