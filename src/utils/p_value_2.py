import streamlit as st
import numpy as np
import prophet as prophet
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import kendalltau, pearsonr, spearmanr

# def correlation_heatmap(data):
#     st.title("Correlaciones")
#     data_numeric = data.select_dtypes(include=[np.number])
    
#     if len(data_numeric) < 2:
#         st.warning("No se puede calcular la correlación: El conjunto de datos numérico tiene menos de 2 filas.")
#     else:
#         corr = data_numeric.corr()
#         fig, ax = plt.subplots(figsize=(30,20))
#         sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
#         st.pyplot(fig)
        
#         p_values_matrix = correlation_pvalue_matrix(data_numeric) 

#         st.divider()

#         st.subheader("Variables con Correlación Significativa (p ≤ 0.05)")
#         st.info("Si la columna p_value es ")
#         # Transformamos la matriz en una lista de pares para que sea legible
#         significant_list = []
#         cols = p_values_matrix.columns
        
#         # Recorremos solo la mitad superior de la matriz para no repetir (A-B y B-A)
#         # y saltamos la diagonal (A-A siempre es p=0)
#         for i in range(len(cols)):
#             for j in range(i + 1, len(cols)):
#                 p_val = p_values_matrix.iloc[i, j]
#                 if p_val <= 0.05:
#                     var1 = cols[i]
#                     var2 = cols[j]
#                     correlation_val = corr.iloc[i, j]
#                     significant_list.append({
#                         "Variable 1": var1,
#                         "Variable 2": var2,
#                         "p-value": round(p_val, 4),
#                         "Correlación": round(correlation_val, 2)
#                     })

#         if significant_list:
#             df_signif = pd.DataFrame(significant_list)
#             # Ordenar por los más significativos primero
#             df_signif = df_signif.sort_values("p-value")
#             st.dataframe(df_signif, use_container_width=True)
#         else:
#             st.info("No se encontraron correlaciones estadísticamente significativas.")

# def correlation_pvalue_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
#     """
#     Computes a matrix of p-values for pairwise correlations.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Numerical dataframe.
#     method : str
#         Correlation method: 'pearson', 'spearman', or 'kendall'.

#     Returns
#     -------
#     pd.DataFrame
#         Matrix of p-values.
#     """
#     cols = df.columns
#     pvals = pd.DataFrame(index=cols, columns=cols, dtype=float)

#     for c1 in cols:
#         for c2 in cols:
#             if method == "pearson":
#                 _, p = pearsonr(df[c1], df[c2])
#             elif method == "spearman":
#                 _, p = spearmanr(df[c1], df[c2])
#             elif method == "kendall":
#                 _, p = kendalltau(df[c1], df[c2])
#             else:
#                 raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")
            
#             pvals.loc[c1, c2] = p

#     return pvals

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr, spearmanr

def correlation_heatmap(data):
    st.title("Correlaciones")
    data_numeric = data.select_dtypes(include=[np.number]).dropna()
    
    if len(data_numeric) < 2:
        st.warning("No hay suficientes datos numéricos.")
    else:
        # 1. Heatmap (Visualización general)
        corr = data_numeric.corr()
        fig, ax = plt.subplots(figsize=(25, 15))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        
        p_values_matrix = correlation_pvalue_matrix(data_numeric) 

        st.divider()
        st.subheader("Variables con Correlación Significativa (p ≤ 0.05)")
        
        st.info("""
        **Guía de interpretación:**
        * **p-value:** Probabilidad de error. Solo se muestran valores que, tras el redondeo, son mayores a 0.0000.
        * **Correlación:** Fuerza de la relación (de -1 a 1).
        """)

        significant_list = []
        cols = p_values_matrix.columns
        
        # Queremos mostrar 4 decimales
        precision = 4
        
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                p_val = p_values_matrix.iloc[i, j]
                
                # PASO 1: Redondeamos primero el valor
                p_val_rounded = round(float(p_val), precision)
                
                # PASO 2: Solo si el redondeado es estrictamente mayor que 0
                # Esto elimina CUALQUIER valor que Streamlit fuera a mostrar como 0.0000
                if 0 < p_val_rounded <= 0.05:
                    significant_list.append({
                        "Variable 1": cols[i],
                        "Variable 2": cols[j],
                        "p-value": p_val_rounded,
                        "Correlación": round(float(corr.iloc[i, j]), 2)
                    })

        if significant_list:
            df_signif = pd.DataFrame(significant_list)
            # Ordenamos por p-value (los más creíbles primero)
            df_signif = df_signif.sort_values("p-value")
            
            # PASO 3: Forzamos a Streamlit a mostrar los decimales para confirmar que no hay ceros
            st.dataframe(
                df_signif, 
                use_container_width=True, 
                hide_index=True
            )
        else:
            st.info("No se encontraron correlaciones con p-value visible (mayores a 0.0000).")

def correlation_pvalue_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Calcula una matriz de p-values para las correlaciones por pares.
    """
    df = df.loc[:, df.std() > 0]
    cols = df.columns
    pvals = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for c1 in cols:
        for c2 in cols:
            if c1 == c2:
                pvals.loc[c1, c2] = 0.0
                continue
            
            try:
                if method == "pearson":
                    _, p = pearsonr(df[c1], df[c2])
                elif method == "spearman":
                    _, p = spearmanr(df[c1], df[c2])
                elif method == "kendall":
                    _, p = kendalltau(df[c1], df[c2])
                pvals.loc[c1, c2] = p
            except:
                pvals.loc[c1, c2] = np.nan
                
    return pvals