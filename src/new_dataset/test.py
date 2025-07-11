import time
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from dlordinal.metrics import minimum_sensitivity
from dlordinal.metrics import accuracy_off1

from lib.utils import load

def main():
	file = {"pca_binary", "under_pca_binary", "pca_multiclass"}
	clf  = None

	print("clasificador,dataset,n patrones,n caracteristicas,accuracy,tiempo,ms,f1")

	for train_file in file:
		X, y = load('../dataset/' + train_file + '.npz')

		X_train, X_test, y_train, y_test = train_test_split(
		 	X, y, test_size = 0.25, random_state = 1
		)

		for i in range(1):
			if i == 0: clf = DecisionTreeClassifier()
			elif i == 1: clf = RandomForestClassifier()
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
			f1_train       = accuracy_off1(y_train, y_pred_train)

			# Evaluar test
			accuracy_test = accuracy_score(y_test, y_pred_test)
			ms_test       = minimum_sensitivity(y_test, y_pred_test)
			f1_test       = accuracy_off1(y_test, y_pred_test)

			print(f"{i},{train_file},{X.shape},{accuracy_train:.3f},{accuracy_test:.3f},{tiempo:.3f},{ms_train:.3f},{f1_train:.3f},{ms_test:.3f},{f1_test:.3f}")

if __name__ == "__main__":
    main()
