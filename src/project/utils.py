import click
import numpy as np

# clasificadores
from sklearn.tree import DecisionTreeClassifier

# matriz de confusion
import matplotlib.pyplot as plt
import seaborn as sns

def load(filename):
	data = np.load(filename)

	return data['X'], data['y']

def selectParams(option):
	param = None

	if option == "decisiontree":
		param = {
			"max_depth"         : np.logspace(0, 2, num = 7, base = 10).astype(int),
			"min_samples_split" : np.logspace(1, 6, num = 4, base = 2).astype(int),
			"min_samples_leaf"  : np.logspace(0, 6, num = 4, base = 2).astype(int),
			"criterion"         : ["gini", "entropy"]
		}
	else:
		raise click.UsageError(f"La opcion '{option}' no se encuentra en la lista.")

	return param

def selectModel(option, random_state):
	model = None

	if option == "decisiontree":
		model = DecisionTreeClassifier(random_state = random_state)
	else:
		raise click.UsageError(f"La opcion '{option}' no se encuentra en la lista.")

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
