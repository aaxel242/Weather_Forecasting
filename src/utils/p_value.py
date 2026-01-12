import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr, spearmanr

def correlation_heatmap(data):
    """
    Muestra matriz de correlaciones (heatmap) y tabla de correlaciones significativas.
    Parámetros: data (DataFrame). Retorna: None (renderiza gráficos en Streamlit).
    """
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
                width="stretch", 
                hide_index=True
            )
        else:
            st.info("No se encontraron correlaciones con p-value visible (mayores a 0.0000).")

def correlation_pvalue_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Calcula matriz de p-values para correlaciones por pares (Pearson, Spearman, Kendall).
    Parámetros: df (DataFrame numérico), method (str tipo correlación).
    Retorna: DataFrame matriz de p-values.
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
