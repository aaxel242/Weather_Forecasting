import json
import pandas as pd

def cargar_json(ruta):
    """Lee un archivo JSON probando distintas codificaciones."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(ruta, "r", encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError):
            continue
        except json.JSONDecodeError as e:
            print(f"Error de formato en {ruta}: {e}")
            return None

    print(f"No se pudo leer el archivo: {ruta}")
    return None

def jsons_a_csv(lista_jsons, salida_csv):
    """Convierte múltiples archivos JSON a un único CSV."""
    datos = []

    for archivo in lista_jsons:
        print(f"Procesando: {archivo}")
        contenido = cargar_json(archivo)

        if contenido is None:
            continue

        if isinstance(contenido, list):
            datos.extend(contenido)
        elif isinstance(contenido, dict):
            datos.append(contenido)

    if not datos:
        print("No se encontraron datos válidos.")
        return None

    df = pd.DataFrame(datos)
    df.to_csv(salida_csv, index=False, encoding="utf-8")

    print(f"CSV generado: {salida_csv}")
    print(f"Filas: {len(df)} | Columnas: {len(df.columns)}")
    return df


if __name__ == "__main__":

    folder = "path"

    archivos = [
        f"{folder}/2024-01-01 a 2024-06-30.txt",
        f"{folder}/2024-07-01 a 2024-12-31.txt",
        f"{folder}/2025-01-01 a 2025-06-30.txt",
        f"{folder}/2025-07-01 a 2025-11-30.txt",
    ]

    salida = "datos_climaticos_completos.csv"
    jsons_a_csv(archivos, salida)