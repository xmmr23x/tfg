import click
import os.path

from lib.utils import load
from lib.utils import resampling
from lib.utils import save

@click.command()
@click.option(
	"-d", "--dataset",
	default = None, type = str, required = True,
	help = "Name of the file with training data.",
)
@click.option(
	"-t", "--train",
	default = None, type = str, required = True,
	help = "Name of the file with new train dataset.",
)
@click.option(
	"-T", "--test",
	default = None, type = str, required = True,
	help = "Name of the file with new test dataset.",
)
@click.option(
	"-u", "--undersampling",
	default = False, type = bool, required = False, is_flag=True,
	help = "Name of the file with new dataset.",
)
def main(
	dataset: str,
	test: str,
	train: str,
	undersampling: bool
):
	if not os.path.isfile(dataset):
		raise click.UsageError(f"El archivo '{dataset}' no existe.")

	if not os.path.isdir(os.path.dirname(test)):
		raise click.UsageError(f"El directorio '{os.path.dirname(test)}' no existe.")

	if not os.path.isdir(os.path.dirname(train)):
		raise click.UsageError(f"El directorio '{os.path.dirname(train)}' no existe.")

	print("Cargando datos...")
	X, y = load(dataset)

	print("Reduciendo la dimensionalidad...")
	X_train, X_test, y_train, y_test = resampling(X, y, u = undersampling)

	print("Guardando datos...")
	save(train, X_train, y_train)
	save(test, X_test, y_test)

if __name__ == "__main__":
	main()
