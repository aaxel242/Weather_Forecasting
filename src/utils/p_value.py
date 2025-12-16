import streamlit as st
import pandas as pd
import numpy as np
import prophet as prophet
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import kendalltau, pearsonr, spearmanr

def correlation_heatmap(data):
        # 2. Correlaciones
    with st.expander("🔗 Correlaciones"):
        st.subheader("🔗 Correlaciones")
        data_numeric = data.select_dtypes(include=["float", "int"])
        
        if len(data_numeric) < 2:
            st.warning("⚠️ No se puede calcular la correlación: El conjunto de datos numérico tiene menos de 2 filas.")
        else:
            corr = data_numeric.corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            
            p_values = correlation_pvalue_matrix(data_numeric) 
            st.write("Variables con correlación significativa (p <= 0.05):")
            st.write(p_values)

def correlation_pvalue_matrix(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes a matrix of p-values for pairwise correlations.

    Parameters
    ----------
    df : pd.DataFrame
        Numerical dataframe.
    method : CorrelationMethods
        Correlation method.

    Returns
    -------
    pd.DataFrame
        Matrix of p-values.
    """
    p_values_final = []
    method = "pearson"
    cols = df.columns
    pvals = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for c1 in cols:
        for c2 in cols:
            if method is method:
                _, p = pearsonr(df[c1], df[c2])
                
                st.info(f"{df[c1]} y {df[c2]}")
                if ((_ and p) <= 0.05) and ([c1] != [c2]):
                    p_values_final.append(c1)
                    p_values_final.append(c2)
            elif method is method:
                _, p = spearmanr(df[c1], df[c2])
            else:
                _, p = kendalltau(df[c1], df[c2])
            pvals.loc[c1, c2] = p

    return p_values_final