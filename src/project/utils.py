import click
import numpy as np

from sklearn.tree import DecisionTreeClassifier

def load(filename):
	data = np.load(filename)

	return data['X'], data['y']

def selectParams(option):
	param = None

	if option == "decissiontree":
		param = {
			"max_depth"         : np.logspace(0, 3, num = 10, base = 10).astype(int),
			"min_samples_split" : np.logspace(-3, 0, num = 5, base = 2),
			"min_samples_leaf"  : np.logspace(0, 8, num = 5, base = 2).astype(int),
			"criterion"         : ["gini", "entropy"]
		}
	else:
		raise click.UsageError(f"La opcion '{option}' no se encuentra en la lista.")

	return param

def selectModel(option, random_state):
	model = None

	if option == "decissiontree":
		model = DecisionTreeClassifier(random_state = random_state)
	else:
		raise click.UsageError(f"La opcion '{option}' no se encuentra en la lista.")

	return model
