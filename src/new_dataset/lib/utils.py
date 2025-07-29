import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

# matriz de confusion
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

def load(filename):
	data = np.load(filename)

	return data['X'], data['y']

def save(filename, X, y):
	np.savez(filename, X = X, y = y)

def resampling(X, y, n_components = 5, size = 15000, u = False):
	if u:
		rus  = RandomUnderSampler(sampling_strategy = {0: size, 1: size})
		# rus  = RandomUnderSampler(sampling_strategy = 'majority')
		X, y = rus.fit_resample(X, y)
	"""
	X_train, X_test, y_train, y_test = train_test_split(
	 	X, y, test_size = 0.25, random_state = 1
	)

	pca     = PCA(n_components)
	X_train = pca.fit_transform(X_train)
	X_test  = pca.transform(X_test)

	return X_train, X_test, y_train, y_test
	"""

	pca = PCA(n_components)
	X   = pca.fit_transform(X)

	return X, y

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
