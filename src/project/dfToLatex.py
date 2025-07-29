import pandas as pd

def toLatex(df, salida):

	latex_table = "\\begin{table}[th]\n"
	latex_table += "\t\\centering\n"
	latex_table += "\t\\begin{tabular}{ |c|c|c|c|c|c|c| }\n"
	latex_table += "\t\t\\hline\n"
	latex_table += "\t\t\\rowcolor{LightCyan}\n"
	latex_table += "\t\t & \\multicolumn{3}{c|}{Entrenamiento} & \\multicolumn{3}{c|}{Test} \\\\\n"
	latex_table += "\t\t\\hline\n"
	latex_table += "\t\t\\rowcolor{LightCyan}\n"
	latex_table += "\t\t Estado aleatorio & Acc & MS & F1 & Acc & MS & F1 \\\\\n"
	latex_table += "\t\t\\hline\n"

	for i, row in df.iterrows():
		row_str = " & ".join([str(i)] + [f"{val:.3f}" for val in row]) + " \\\\\n"
		latex_table += f"\t\t{row_str}"

	latex_table += "\t\t\\hline\n"
	latex_table += "\t\\end{tabular}\n"
	latex_table += "\t\\caption{caption}\n"
	latex_table += "\t\\label{label}\n"
	latex_table += "\\end{table}\n"


	with open(salida, 'w') as f:
		f.write(latex_table)