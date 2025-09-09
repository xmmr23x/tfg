import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('resultados/svm_bin.csv', index_col = 0)

# df_clean = df.drop(index=["Mean", "STD"], errors="ignore")

# Convertir de formato ancho a largo
df_long = df.melt(
    value_vars=["acc train", "acc test"],
    var_name="tipo",
    value_name="valor"
)

plt.figure(figsize=(8, 6))

# Violinplot
sns.violinplot(
    x="tipo",
    y="valor",
    data=df_long,
    inner=None,  # quitamos el box interno
    color="skyblue"
)

# Boxplot encima
sns.boxplot(
    x="tipo",
    y="valor",
    data=df_long,
    width=0.2,
    boxprops={"facecolor": "white", "zorder": 2},
    showcaps=True,
    whiskerprops={"linewidth": 2},
    zorder=3
)

plt.title("Distribución de accuracy (train vs test) en SVC", fontsize=14)
plt.show()