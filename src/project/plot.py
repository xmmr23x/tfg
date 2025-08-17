import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('prueba.csv', index_col = 0)

# Supongamos que tu DataFrame se llama df
# Primero eliminamos las filas "Mean" y "STD" si están como índices
df_clean = df.drop(index=["Mean", "STD"], errors="ignore")

# Convertir de formato ancho a largo
df_long = df_clean.melt(
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

plt.title("Distribución de accuracy (train vs test)", fontsize=14)
plt.show()