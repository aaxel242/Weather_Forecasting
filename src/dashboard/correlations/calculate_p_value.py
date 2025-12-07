import pandas as pd
import numpy as np
import prophet as prophet

from scipy.stats import kendalltau, pearsonr, spearmanr

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
    method = "person"
    cols = df.columns
    pvals = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for c1 in cols:
        for c2 in cols:
            if method is method:
                _, p = pearsonr(df[c1], df[c2])
                
                # st.info(f"{df[c1]} y {df[c2]}")
                if ((_ and p) <= 0.05) and ([c1] != [c2]):
                    p_values_final.append(c1)
                    p_values_final.append(c2)
            elif method is method:
                _, p = spearmanr(df[c1], df[c2])
            else:
                _, p = kendalltau(df[c1], df[c2])
            pvals.loc[c1, c2] = p

    return p_values_final