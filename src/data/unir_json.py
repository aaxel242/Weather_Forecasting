import os
import pandas as pd

def unir_json_a_csv(carpeta_raiz, salida_csv):
    """
    Recorre todas las subcarpetas dentro de carpeta_raiz y une todos los JSON en un único CSV.
    Parámetros: carpeta_raiz (str ruta), salida_csv (str ruta archivo salida).
    Retorna: DataFrame con datos unificados o None si hay error.
    """
    dataframes = []
    codificaciones = ["utf-8", "latin1", "windows-1252"]

    # Recorrer todas las subcarpetas y archivos
    for root, _, files in os.walk(carpeta_raiz):
        for archivo in files:
            if archivo.endswith(".txt") or archivo.endswith(".json"):
                ruta = os.path.join(root, archivo)

                # Intentar leer con varias codificaciones
                df = None
                for cod in codificaciones:
                    try:
                        with open(ruta, "r", encoding=cod) as f:
                            df = pd.read_json(f)
                        break
                    except Exception:
                        continue

                if df is not None:
                    dataframes.append(df)
                else:
                    print(f"⚠️ No se pudo leer el archivo: {ruta}")

    # Concatenar todos los DataFrames
    if dataframes:
        df_final = pd.concat(dataframes, ignore_index=True)
        df_final.to_csv(salida_csv, index=False)
        print(f"✅ Dataset unificado guardado en: {salida_csv}")
        return df_final
    else:
        print("⚠️ No se encontraron datos válidos.")
        return None
