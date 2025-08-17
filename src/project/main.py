import os.path
import time
import warnings

from utils import *
from dfToLatex import toLatex

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

@click.command()
@click.option(
	'-d', '--dataset',
	default = None, type = str, required = True,
	help = 'Nombre del archivo del dataset de entrenamiento.',
)
@click.option(
	'-o', '--option',
	default = 'dt', type = str, required = False,
	help = 'Indica el modelo de entrenamiento que se va a usar. [dt, knn, rf]',
)
@click.option(
	'-s', '--seeds',
	default = 10, type = int, required = False,
	help = 'Numero de semillas a usar.',
)
@click.option(
	'-c', '--crossval',
	default = 5, type = int, required = False,
	help = 'Numero de subconjuntos a usar en la validación cruzada.',
)
@click.option(
	'-w', '--write',
	default = None, type = str, required = False,
	help = 'Nombre del archivo de salida donde se guardan los resultados.',
)
@click.option(
	'-m', '--matrix',
	default = None, type = str, required = False,
	help = 'Indica si se desea visualizar la matriz de confusión. [test, train]',
)
@click.option(
	'-l', '--latex',
	default = None, type = str, required = False,
	help = 'Nombre del archivo de salida donde se guardan los resultados en formato latex.',
)
@click.option(
	'-i', '--it',
	default = 15, type = int, required = False,
	help = 'Número de combinaciones de parámetros que se muestrean.',
)
def main(
	dataset : str,
	option  : str,
	seeds   : int,
	crossval: int,
	write   : str,
	matrix  : str,
	latex   : str,
	it      : int
):
	options = ['dt', 'knn', 'rf', 'ridge', 'svm', 'mlp', 'lgbm']
	matrix_ = ['test', 'train']
	tiempo_total = 0

	if matrix and matrix not in matrix_:
		raise click.UsageError(f'La opcion {matrix} no se encuentra en la lista.')

	if option not in options:
		raise click.UsageError(f'La opcion {option} no se encuentra en la lista.')

	if not os.path.isfile(dataset):
		raise click.UsageError(f'El archivo {dataset} no existe.')

	if write:
		dir_name = os.path.dirname(write)

		if dir_name and not os.path.isdir(dir_name):
			raise click.UsageError(f'El directorio {dir_name} no existe. No se pueden guardar los resultados.')

	if latex:
		dir_name = os.path.dirname(latex)

		if dir_name and not os.path.isdir(dir_name):
			raise click.UsageError(f'El directorio {dir_name} no existe. No se pueden exportar los resultados a una tabla latex.')

	warnings.filterwarnings("ignore", message = ".*does not have valid feature names.*")

	X_train = X_test = y_train = y_test = y_pred_train = y_pred_test = None

	param    = selectParams(option)
	score    = pd.DataFrame(columns = ['acc train', 'ms train', 'f1 train', 'acc test', 'ms test', 'f1 test'])
	X, y     = load(dataset)
	X        = norm(X, y)

	print(X.shape)

	for random_state in range(seeds):
		print(f'##################### SEED {random_state} #####################')

		model = selectModel(option, random_state)

		X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size = 0.25, random_state = random_state
		)

		crossval = cv(y_train, crossval)
		clf = GridSearchCV(model, param, cv = crossval, n_jobs = -1)
		# clf = RandomizedSearchCV(model, param, n_iter = it, n_jobs = -1, cv = crossval, random_state = random_state)

		total_combinations = np.prod([len(v) for v in param.values()])


		inicio = time.time()

		clf.fit(X_train, y_train)

		tiempo        = time.time() - inicio
		tiempo_total += tiempo


		y_pred_train = clf.predict(X_train)
		y_pred_test  = clf.predict(X_test)

		print(clf.best_params_)
		print(f'Tiempo de entrenamiento: {tiempo}')

		score.loc[random_state] = scores(y_train, y_pred_train, y_test, y_pred_test)

		print(score.loc[random_state])

	score.loc['Mean'] = score.mean()
	score.loc['STD'] = score.head(seeds).std()

	print(score)
	print(f'Tiempo total de entrenamiento: {tiempo_total}')

	if write:
		score.to_csv(write)

	if matrix:
		if matrix == 'test':
			confussionMatrix(y_test, y_pred_test)
		if matrix == 'train':
			confussionMatrix(y_train, y_pred_train)

	if latex:
		toLatex(score, latex)

if __name__ == '__main__':
	main()
