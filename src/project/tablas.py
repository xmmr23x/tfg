from dfToLatex import toLatex

import pandas as pd

df = pd.read_csv('prueba.csv', index_col = 0)

print(df)

toLatex(df, 'rf_bin.tex')
