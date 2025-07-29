from lib.utils import load
from lib.utils import resampling
from lib.utils import save

import pandas as pd

def main():
	X, y        = load('../dataset/bodmas.npz')
	metadata    = pd.read_csv('../dataset/bodmas_metadata.csv')
	mw_category = pd.read_csv('../dataset/bodmas_malware_category.csv')

	# Incluimos los valores de 'category' en metadata cuando coinciden los valoes de 'sha'
	mw_category = metadata.merge(mw_category, on = 'sha', how = 'left')

	# Rellenamos los huecos como software benigno
	mw_category['category'] = mw_category['category'].fillna('benign')

	# Eliminamos todas las columnas excepto 'category'
	mw_category = mw_category['category']

	# Codificamos las categorias de malware
	category = {
		'benign': 0, 'trojan': 1, 'worm': 2, 'backdoor': 3,
		'downloader': 4, 'informationstealer': 5, 'dropper': 6,
		'ransomware': 7, 'rootkit': 9, 'cryptominer': 9, 'pua': 9,
		'exploit': 9, 'virus': 8, 'p2p-worm': 2, 'trojan-gamethief': 1
	}

	# print(mw_category.value_counts())

	mw_category = mw_category.map(category)

	y = mw_category.to_numpy()

	save('../dataset/menos_clases.npz', X, y)

if __name__ == '__main__':
	main()
