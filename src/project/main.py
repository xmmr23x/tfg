import os.path

from utils import *
from dfToLatex import toLatex

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from dlordinal.metrics import minimum_sensitivity
from dlordinal.metrics import accuracy_off1

@click.command()
@click.option(
	"-d", "--dataset",
	default = None, type = str, required = True,
	help = "Nombre del archivo del dataset de entrenamiento.",
)
@click.option(
	"-o", "--option",
	default = "decisiontree", type = str, required = False,
	help = "Indica el modelo de entrenamiento que se va a usar. [decisiontree, knn, randomforest]",
)
@click.option(
	"-s", "--seeds",
	default = 10, type = int, required = False,
	help = "Numero de semillas a usar.",
)
@click.option(
	"-c", "--crossval",
	default = 5, type = int, required = False,
	help = "Numero de subconjuntos a usar en la validación cruzada.",
)
@click.option(
	"-w", "--write",
	default = None, type = str, required = False,
	help = "Nombre del archivo de salida donde se guardan los resultados.",
)
@click.option(
	"-m", "--matrix",
	default = None, type = str, required = False,
	help = "Indica si se desea visualizar la matriz de confusión. [test, train]",
)
@click.option(
	"-l", "--latex",
	default = None, type = str, required = False,
	help = "Nombre del archivo de salida donde se guardan los resultados en formato latex.",
)
def main(
	dataset: str,
	option: int,
	seeds: int,
	crossval: int,
	write: str,
	matrix: str,
	latex: str
):
	options = ['decisiontree', 'knn', 'randomforest']
	matrix_ = ['test', 'train']

	if matrix and matrix not in matrix_:
		raise click.UsageError(f"La opcion '{matrix}' no se encuentra en la lista.")

	if option not in options:
		raise click.UsageError(f"La opcion '{option}' no se encuentra en la lista.")

	if not os.path.isfile(dataset):
		raise click.UsageError(f"El archivo '{dataset}' no existe.")

	if write:
		dir_name = os.path.dirname(write)

		if dir_name and not os.path.isdir(dir_name):
			raise click.UsageError(f"El directorio '{dir_name}' no existe. No se pueden guardar los resultados.")

	if latex:
		dir_name = os.path.dirname(latex)

		if dir_name and not os.path.isdir(dir_name):
			raise click.UsageError(f"El directorio '{dir_name}' no existe. No se pueden exportar los resultados a una tabla latex.")

	X_train = X_test = y_train = y_test = y_pred_train = y_pred_test = None

	param    = selectParams(option)
	score    = pd.DataFrame(columns = ['acc train', 'ms train', 'f1 train', 'acc test', 'ms test', 'f1 test'])
	X, y     = load(dataset)
	X        = norm(X, y)
	crossval = cv(y, crossval)

	for random_state in range(seeds):
		print(f"##################### SEED {random_state} #####################")
		model = selectModel(option, random_state)

		X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size = 0.25, random_state = random_state
		)

		clf = GridSearchCV(model, param, n_jobs = -1, cv = crossval)
		clf.fit(X_train, y_train)

		y_pred_train = clf.predict(X_train)
		y_pred_test  = clf.predict(X_test)

		print(clf.best_params_)

		score.loc[random_state] = [
			accuracy_score(y_train, y_pred_train),
			minimum_sensitivity(y_train, y_pred_train),
			accuracy_off1(y_train, y_pred_train),
			accuracy_score(y_test, y_pred_test),
			minimum_sensitivity(y_test, y_pred_test),
			accuracy_off1(y_test, y_pred_test)
		]

		print(score.loc[random_state])

	score.loc["Mean"] = score.mean()
	score.loc["STD"] = score.head(seeds).std()


	print(score)

	if write:
		score.to_csv(write)

	if matrix:
		if matrix == "test":
			confussionMatrix(y_test, y_pred_test)
		if matrix == "train":
			confussionMatrix(y_train, y_pred_train)

	if latex:
		toLatex(score, latex)

if __name__ == '__main__':
	main()
