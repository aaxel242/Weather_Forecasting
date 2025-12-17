# Tabla campo / significado

| Campo         | Significado                                                                                   |
| ------------- | --------------------------------------------------------------------------------------------- |
| **tmed**      | Temperatura media del día (°C), calculada normalmente como promedio entre la máxima y mínima. |
| **prec**      | Precipitación del día (mm o litros/m²). Si es 0 → no llovió.                                  |
| **tmin**      | Temperatura mínima registrada en el día (°C).                                                 |
| **horatmin**  | Hora en la que ocurrió la temperatura mínima.                                                 |
| **tmax**      | Temperatura máxima registrada en el día (°C).                                                 |
| **horatmax**  | Hora en la que ocurrió la temperatura máxima.                                                 |
| **dir**       | Dirección del viento predominante (grados). Ej: 0° = Norte, 90° = Este, 180° = Sur, etc.      |
| **velmedia**  | Velocidad media del viento durante el día (m/s o km/h según estación).                        |
| **racha**     | Racha máxima de viento registrada en el día.                                                  |
| **horaracha** | Hora en la que se registró la racha máxima de viento.                                         |
| **hrMedia**   | Humedad relativa media del día (%).                                                           |
| **hrMax**     | Máxima humedad relativa del día (%).                                                          |
| **horaHrMax** | Hora en la que ocurrió la humedad máxima.                                                     |
| **hrMin**     | Mínima humedad relativa del día (%).                                                          |
| **horaHrMin** | Hora en la que ocurrió la humedad mínima.                                                     |

| Columna          | Qué es               | Unidad         | Representación |
| ---------------- | -------------------- | -------------- | -------------- |
| date             | Fecha                | YYYY-MM-DD     | Tiempo         |
| tmed             | Temp. media diaria   | °C             | Numérico       |
| tmin             | Temp. mínima         | °C             | Numérico       |
| tmax             | Temp. máxima         | °C             | Numérico       |
| prec             | Precipitación        | mm             | Numérico       |
| dir              | Dirección del viento | grados (0–360) | Numérico       |
| velmedia         | Vel. media viento    | km/h           | Numérico       |
| racha            | Racha máx. viento    | km/h           | Numérico       |
| hrmedia          | Humedad media        | %              | Numérico       |
| hrmax            | Humedad máxima       | %              | Numérico       |
| hrmin            | Humedad mínima       | %              | Numérico       |
| cloudcover       | Nubosidad            | %              | Numérico       |
| surface_pressure | Presión              | hPa            | Numérico       |
| dewpoint_2m      | Punto de rocío       | °C             | Numérico       |
| estacion_final   | Estación del año     | texto          | Categórica     |
