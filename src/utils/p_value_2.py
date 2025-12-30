import streamlit as st
import numpy as np
import prophet as prophet
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import kendalltau, pearsonr, spearmanr

def correlation_heatmap(data):
    st.title("🔗 Correlaciones")
    data_numeric = data.select_dtypes(include=[np.number])
    
    if len(data_numeric) < 2:
        st.warning("⚠️ No se puede calcular la correlación: El conjunto de datos numérico tiene menos de 2 filas.")
    else:
        corr = data_numeric.corr()
        fig, ax = plt.subplots(figsize=(30,20))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        
        p_values = correlation_pvalue_matrix(data_numeric) 
        st.write("Variables con correlación significativa (p <= 0.05):")
        st.write(p_values)

def correlation_pvalue_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Computes a matrix of p-values for pairwise correlations.

    Parameters
    ----------
    df : pd.DataFrame
        Numerical dataframe.
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall'.

    Returns
    -------
    pd.DataFrame
        Matrix of p-values.
    """
    cols = df.columns
    pvals = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for c1 in cols:
        for c2 in cols:
            if method == "pearson":
                _, p = pearsonr(df[c1], df[c2])
            elif method == "spearman":
                _, p = spearmanr(df[c1], df[c2])
            elif method == "kendall":
                _, p = kendalltau(df[c1], df[c2])
            else:
                raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")
            
            pvals.loc[c1, c2] = p

    return pvals