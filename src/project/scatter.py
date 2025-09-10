import pandas as pd
from aeon.visualisation import plot_pairwise_scatter
from aeon.benchmarking.results_loaders import get_estimator_results_as_array
import numpy as np
import matplotlib.pyplot as plt

df_rf    = pd.read_csv("resultados/rf_multi.csv", index_col=0)
df_lgbm  = pd.read_csv("resultados/knn_multi.csv", index_col=0)
rf_acc   = np.array(df_rf["acc test"].values)
lgbm_acc = np.array(df_lgbm["acc test"].values)
results  = [rf_acc, lgbm_acc]

methods = ["RF", "KNN"]

plot = plot_pairwise_scatter(rf_acc, lgbm_acc, methods[0], methods[1])
plt.show()
