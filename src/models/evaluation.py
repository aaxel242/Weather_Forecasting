import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

def evaluate_classification(y_true, y_pred):
    # Forzamos conversión a entero para evitar el error de '0.0' vs 0
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    
    return {
        "Precisión Global (Accuracy)": f"{accuracy_score(y_true, y_pred):.2%}",
        "Precisión (Evitar Falsos Positivos)": f"{precision_score(y_true, y_pred, zero_division=0):.2%}",
        "Recall (Detectar Lluvia Real)": f"{recall_score(y_true, y_pred, zero_division=0):.2%}",
        "F1-Score (Balance)": f"{f1_score(y_true, y_pred, zero_division=0):.2%}"
    }