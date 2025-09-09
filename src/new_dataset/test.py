import time
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from dlordinal.metrics import minimum_sensitivity

from lib.utils import load
from lib.utils import confussionMatrix

def main():
	# file = {"pca_binary", "under_pca_binary", "pca_multiclass"}
	file = {"menos_clases"}
	clf  = None

	print("clasificador,dataset,n patrones,n caracteristicas,tiempo,accuracy train,ms train,f1 train,accuracy test,ms test,f1 test")

	for train_file in file:
		X, y = load('../dataset/' + train_file + '.npz')

		X_train, X_test, y_train, y_test = train_test_split(
		 	X, y, test_size = 0.25, random_state = 1
		)

		for i in range(1):
			if i == 0: clf = RidgeClassifier(class_weight = "balanced")
			elif i == 1: clf = RandomForestClassifier(class_weight = "balanced")
			else: clf = KNeighborsClassifier()

			# Entrenar el modelo
			inicio = time.time()
			clf.fit(X_train, y_train)
			tiempo = time.time() - inicio

			# Predecir sobre el conjunto de prueba
			y_pred_train = clf.predict(X_train)
			y_pred_test  = clf.predict(X_test)

			# Evaluar entrenamiento
			accuracy_train = accuracy_score(y_train, y_pred_train)
			ms_train       = minimum_sensitivity(y_train, y_pred_train)
			f1_train       = f1_score(y_train, y_pred_train, average = 'weighted')

			# Evaluar test
			accuracy_test = accuracy_score(y_test, y_pred_test)
			ms_test       = minimum_sensitivity(y_test, y_pred_test)
			f1_test       = f1_score(y_test, y_pred_test, average = 'weighted')

			print(f"{i},{train_file},{X.shape},{tiempo:.5f},{accuracy_train:.5f},{ms_train:.5f},{f1_train:.5f},{accuracy_test:.5f},{ms_test:.5f},{f1_test:.5f}")

		# confussionMatrix(y_test, y_pred_test)

if __name__ == "__main__":
    main()
