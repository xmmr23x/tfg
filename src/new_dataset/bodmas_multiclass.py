from lib.utils import load
from lib.utils import resampling
from lib.utils import save

import pandas as pd

def main():
	X, y        = load('bodmas/bodmas.npz')
	metadata    = pd.read_csv('bodmas/bodmas_metadata.csv')
	mw_category = pd.read_csv('bodmas/bodmas_malware_category.csv')

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
		'ransomware': 7, 'rootkit': 8, 'cryptominer': 9, 'pua': 10,
		'exploit': 11, 'virus': 12, 'p2p-worm': 13, 'trojan-gamethief': 14
	}

	mw_category = mw_category.map(category)

	y = mw_category.to_numpy()

	save('bodmas/bodmas_multiclass.npz', X, y)

if __name__ == '__main__':
	main()
