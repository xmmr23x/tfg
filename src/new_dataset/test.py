import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from dlordinal.metrics import minimum_sensitivity
from dlordinal.metrics import accuracy_off1

from lib.utils import load

def main():
	file = {"pca_binary", "resampling_binary", "pca_multiclass"}
	clf  = None

	print("clasificador,dataset,n patrones,n caracteristicas,accuracy,tiempo,ms,f1")

	for i in range(3):
		if i == 0: clf = DecisionTreeClassifier()
		elif i == 1: clf = RandomForestClassifier()
		else: clf = KNeighborsClassifier()

		for train_file in file:

			X_train, y_train = load('../dataset/' + train_file + '_train.npz')
			X_test, y_test = load('../dataset/' + train_file + '_test.npz')

			# Entrenar el modelo
			inicio = time.time()
			clf.fit(X_train, y_train)
			tiempo = time.time() - inicio

			# Predecir sobre el conjunto de prueba
			y_pred = clf.predict(X_test)

			# Evaluar
			accuracy = accuracy_score(y_test, y_pred)
			ms       = minimum_sensitivity(y_test, y_pred)
			f1       = accuracy_off1(y_test, y_pred)

			print(f"{i},{train_file},{X_train.shape},{accuracy:.3f},{tiempo:.3f},{ms:.3f},{f1:.3f}")

if __name__ == "__main__":
    main()
