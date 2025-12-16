# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor

# def predecir_clima(data: pd.DataFrame, fecha, rango: int):
#     """
#     Función de ejemplo para predecir el clima.
    
#     Parámetros:
#     - data: DataFrame con los datos climáticos.
#     - fecha: fecha inicial seleccionada en el sidebar.
#     - rango: número de días a predecir.

#     Retorna:
#     - Un DataFrame o diccionario con las predicciones.
#     """

#     # Aquí deberías aplicar tu modelo real de predicción.
#     # De momento, devolvemos un resultado simulado para ilustrar el flujo.
    
#     resultados = []
#     for i in range(rango):
#         dia = pd.to_datetime(fecha) + pd.Timedelta(days=i)
#         resultados.append({
#             "Fecha": dia.strftime("%Y-%m-%d"),
#             "Predicción": "Soleado ☀️" if i % 2 == 0 else "Nublado ☁️"
#         })

#     return pd.DataFrame(resultados)

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def load_model(model_path: str) -> RandomForestRegressor:
    """
    Load a previously saved ML model using joblib.

    Parameters
    ----------
    model_path : str
        Path to the stored model (.pkl file).

    Returns
    -------
    model
        Loaded ML model.
    """
    return joblib.load(model_path)


def predict_with_model(model, new_features: pd.DataFrame | dict):
    """
    Use a pre-trained model to generate predictions.

    Parameters
    ----------
    model : object
        Pre-trained ML model loaded via joblib.
    new_features : pd.DataFrame or dict
        New feature data to predict on.

    Returns
    -------
    array-like
        Predictions for the given feature set.
    """
    if isinstance(new_features, dict):
        new_features = pd.DataFrame([new_features])

    return model.predict(new_features)
