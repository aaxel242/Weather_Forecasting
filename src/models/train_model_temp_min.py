import joblib
import pandas as pd
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_temp_min_model(features, target, model_path="src/models/model_temp_min_rf.pkl"):
    """
    Entrena un modelo de regresión para la temperatura mínima a 7 días.
    """
    # Dividir datos (80% tren, 20% test)
    split = int(len(features) * 0.8)
    X_train, X_test = features.iloc[:split], features.iloc[split:]
    y_train, y_test = target.iloc[:split], target.iloc[split:]

    # Creamos el pipeline de regresión
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model.fit(X_train, y_train)
    
    # Guardamos tu modelo (.pkl)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    predictions = pd.Series(model.predict(X_test), index=y_test.index)
    return model, predictions