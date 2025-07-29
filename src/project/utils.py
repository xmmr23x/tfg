import click

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

# clasificadores
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# matriz de confusion
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

def load(filename):
	data = np.load(filename)

	return data['X'], data['y']

def selectParams(option):
	param = None

	if option == "decisiontree":
		param = {
			"max_depth"         : np.logspace(1, 2, num = 4, base = 10).astype(int),
			"min_samples_split" : np.logspace(1, 6, num = 4, base = 2).astype(int),
			"min_samples_leaf"  : np.logspace(0, 6, num = 4, base = 2).astype(int),
			"criterion"         : ["gini", "entropy"]
		}
	elif option == "randomforest":
		param = {
			'n_estimators': np.linspace(2100, 2300, 5, dtype = int),
			'max_depth': [170, 180, 190, 200, 210, 220],
			'min_samples_split': [2, 3, 4],
			'min_samples_leaf': [2, 3, 4, 5]
		}
	elif option == "knn":
		param = {
			
		}

	return param

def selectModel(option, random_state):
	model = None

	if option == "decisiontree":
		model = DecisionTreeClassifier(random_state = random_state, class_weight = "balanced")
	elif option == "randomforest":
		model = RandomForestClassifier(random_state = random_state, class_weight = "balanced")
	elif option == "knn":
		model = KNeighborsClassifier()

	return model

def confussionMatrix(y_true, y_pred, title = 'Matriz de Confusión'):
	cm = confusion_matrix(y_true, y_pred)

	plt.figure(figsize=(6, 5))
	sns.heatmap(
		cm,
		annot = True,
		fmt = 'd',
		cmap = 'Blues',
		xticklabels = np.unique(y_true),
		yticklabels = np.unique(y_true)
	)

	plt.xlabel('Predicción')
	plt.ylabel('Real')
	plt.title(title)

	plt.show()

def norm(X, y):
	scaler = MinMaxScaler()

	return scaler.fit_transform(X)

def cv(y, crossval):
	y_ = min(pd.DataFrame(y).value_counts())

	if y_ < crossval:
		return y_

	return crossval
