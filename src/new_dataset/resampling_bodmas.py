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
	"-n", "--new_dataset",
	default = None, type = str, required = True,
	help = "Name of the file with new dataset.",
)
@click.option(
	"-u", "--undersampling",
	default = False, type = bool, required = False, is_flag=True,
	help = "Name of the file with new dataset.",
)
def main(
	dataset: str,
	new_dataset: str,
	undersampling: bool
):
	if not os.path.isfile(dataset):
		raise click.UsageError(f"El archivo '{dataset}' no existe.")

	if not os.path.isdir(os.path.dirname(new_dataset)):
		raise click.UsageError(f"El directorio '{os.path.dirname(train)}' no existe.")

	print("Cargando datos...")
	X, y = load(dataset)

	print("Reduciendo la dimensionalidad...")
	X, y = resampling(X, y, u = undersampling)

	print("Guardando datos...")
	save(new_dataset, X, y)

if __name__ == "__main__":
	main()
