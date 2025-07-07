import os.path
import pandas as pd

from utils import *

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

@click.command()
@click.option(
	"-t", "--train",
	default = None, type = str, required = True,
	help = "Nombre del archivo del dataset de entrenamiento.",
)
@click.option(
	"-T", "--test",
	default = None, type = str, required = True,
	help = "Nombre del archivo del dataset de test.",
)
@click.option(
	"-o", "--option",
	default = "decissiontree", type = str, required = False,
	help = "Indica el modelo de entrenamiento que se va a usar. [decissiontree]",
)
@click.option(
	"-s", "--seeds",
	default = 10, type = int, required = False,
	help = "Numero de semillas a usar.",
)
def main(train: str, test: str, option: int, seeds: int):
	if not os.path.isfile(train):
		raise click.UsageError(f"El archivo '{train}' no existe.")
	if not os.path.isfile(test):
		raise click.UsageError(f"El archivo '{test}' no existe.")

	y_pred_train = None
	y_pred_test  = None
	param        = selectParams(option)
	score        = pd.DataFrame(columns = ['mean_train', 'mean_test'])

	X_train, y_train = load(train)
	X_test, y_test   = load(test)


	for random_state in range(seeds):
		model = selectModel(option, random_state)
		clf   = GridSearchCV(model, param, n_jobs = -1, cv = 5)

		clf.fit(X_train, y_train)

		y_pred_train = clf.predict(X_train)
		y_pred_test  = clf.predict(X_test)

		score.loc[random_state] = [accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test)]

	print(score)

if __name__ == '__main__':
	main()
