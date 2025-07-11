import os.path
import pandas as pd

from utils import *

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

@click.command()
@click.option(
	"-d", "--dataset",
	default = None, type = str, required = True,
	help = "Nombre del archivo del dataset de entrenamiento.",
)
@click.option(
	"-o", "--option",
	default = "decisiontree", type = str, required = False,
	help = "Indica el modelo de entrenamiento que se va a usar. [decisiontree]",
)
@click.option(
	"-s", "--seeds",
	default = 10, type = int, required = False,
	help = "Numero de semillas a usar.",
)
def main(dataset: str, option: int, seeds: int):
	if not os.path.isfile(dataset):
		raise click.UsageError(f"El archivo '{dataset}' no existe.")

	y_pred_train = None
	y_pred_test  = None
	param        = selectParams(option)
	score        = pd.DataFrame(columns = ['mean_train', 'mean_test'])

	X, y = load(dataset)

	for random_state in range(seeds):
		print(f"-------{random_state}-------")
		model = selectModel(option, random_state)

		X_train, X_test, y_train, y_test = train_test_split(
		 	X, y, test_size = 0.25, random_state = random_state
		)

		clf   = GridSearchCV(model, param, n_jobs = -1, cv = 3)
		clf.fit(X_train, y_train)

		y_pred_train = clf.predict(X_train)
		y_pred_test  = clf.predict(X_test)

		print(clf.best_params_)

		score.loc[random_state] = [accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test)]

		print(score.loc[random_state])

	print(score)

if __name__ == '__main__':
	main()
