import pandas as pd

# Cargar los dos CSV
df1 = pd.read_csv("datos_meteorologicos_barcelona_2024_por_dias.csv")
df2 = pd.read_csv("datos_climaticos_completos.csv")

# Asegurar que las columnas de fecha tengan el mismo formato
df1['date'] = pd.to_datetime(df1['date']).dt.date
df2['fecha'] = pd.to_datetime(df2['fecha']).dt.date

# Fusionar los dos DataFrames por la fecha
df_merged = pd.merge(df1, df2, left_on='date', right_on='fecha', how='outer')

# Guardar en un nuevo CSV
df_merged.to_csv("datos_barcelona_fusionados.csv", index=False)

print("Archivo 'datos_barcelona_fusionados.csv' creado con éxito.")
