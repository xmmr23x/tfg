import click

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import randint

# clasificadores
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

# matriz de confusion
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# metricas
from sklearn.metrics import accuracy_score
from dlordinal.metrics import minimum_sensitivity
from dlordinal.metrics import accuracy_off1

def load(filename):
	data = np.load(filename)

	return data['X'], data['y']

def selectParams(option):
	param = None

	if option == 'dt':
		param = {
			'max_depth'         : np.logspace(1, 2, num = 4, base = 10).astype(int),
			'min_samples_split' : np.logspace(1, 6, num = 4, base = 2).astype(int),
			'min_samples_leaf'  : np.logspace(0, 6, num = 4, base = 2).astype(int),
			'criterion'         : ['gini', 'entropy']
		}
	elif option == 'rf':
		param = {
			#'n_estimators': np.linspace(2100, 2300, 5, dtype = int),
			'max_depth'        : np.logspace(1, 2, num = 4, base = 10).astype(int),
			'min_samples_split': np.logspace(1, 6, num = 4, base = 2).astype(int),
			'min_samples_leaf' : np.logspace(0, 6, num = 4, base = 2).astype(int)
			#'criterion'         : ['gini', 'entropy']
		}
	elif option == 'knn':
		param = {
			'n_neighbors' : np.logspace(0, 6, num = 6, base = 2).astype(int),
			'weights'     : ['uniform', 'distance'],
			'algorithm'   : ['auto', 'ball_tree', 'kd_tree', 'brute'],
			# 'leaf_size'   : np.logspace(0, 6, num = 6, base = 2).astype(int)
		}
	elif option == 'ridge':
		param = {
			'alpha'         : [0.001, 0.01, 0.1, 1.0, 10],
			'solver'        : ['auto', 'lsqr'],
			'fit_intercept' : [True, False]
		}
	elif option == 'svm':
		param = {
			# 'gamma' :  np.logspace(-2, 2, num = 5, base = 10),
			# 'C' : [0.1, 1, 10],
			# 'C' : np.logspace(-1, 1, num = 5, base = 10),
			'kernel' : ['poly', 'sigmoid'],
			# 'degree' : [2, 3, 4]
		}
	elif option == 'mlp':
		param = {
			'hidden_layer_sizes' : [ (50, 50, 50), (75, 75, 75), (100, 100, 100) ],
			# 'hidden_layer_sizes' : [ (100, 100) ],
			# 'activation'         : [ 'relu', 'logistic' ],
			# 'alpha'              : [ 1e-5, 1e-4 ],
			# 'learning_rate_init' : [ 0.001, 0.01 ]
		}
	elif option == 'lgbm':
		param = {
			# 'num_leaves'    : np.logspace(4, 7, num = 4, base = 2).astype(int),
			'n_estimators'  : np.logspace(2, np.log10(500), num = 3).astype(int),
		}

	return param

def selectModel(option, random_state):
	model = None

	if option == 'dt':
		model = DecisionTreeClassifier(random_state = random_state, class_weight = 'balanced')
	elif option == 'rf':
		model = RandomForestClassifier(random_state = random_state, class_weight = 'balanced')
	elif option == 'knn':
		model = KNeighborsClassifier()
	elif option == 'ridge':
		model = RidgeClassifier(random_state = random_state, class_weight = 'balanced', max_iter = 2000)
	elif option == 'svm':
		model = SVC(random_state = random_state, class_weight = 'balanced')
	elif option == 'mlp':
		model = MLPClassifier(random_state = random_state, max_iter = 1000)
	elif option == 'lgbm':
		model = LGBMClassifier(random_state = random_state, class_weight = 'balanced', verbose = -1)

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

def scores(y_train, y_pred_train, y_test, y_pred_test):
	return [
		accuracy_score(y_train, y_pred_train),
		minimum_sensitivity(y_train, y_pred_train),
		accuracy_off1(y_train, y_pred_train),
		accuracy_score(y_test, y_pred_test),
		minimum_sensitivity(y_test, y_pred_test),
		accuracy_off1(y_test, y_pred_test)
	]
