Añadimos la columna de bin_prep para:

    - Para que a la hora de hacer el modelo sea mas facil de predecir
    - Tambien para decir que en cualquier caso que la prep sea > 0 siempre llovera aunque sea un poco
    - Y para decir si tienes que llevar paraguas o no, creo que es mas sencillo hacerlo binario que no darle un porcentage al usuario sobre cuanta lluvia va a caer

GENERAR FEATURES:

    - EXPLICACIÓN SENCILLA (nivel básico):
        - Esta explicación es para alguien que empieza:

        - Lags: Son los valores de días anteriores. Los usamos porque el clima de hoy influye en el clima de mañana.

        - Rolling windows: Son promedios de varios días. Sirven para ver tendencias y no solo valores sueltos.

        - Mes, día del año, semana: El clima cambia según la época del año, así que añadimos estas columnas para que el modelo lo entienda.

        - Interpolación: Cuando creamos lags y rolling aparecen huecos. Los rellenamos suavemente para no perder datos.

        - Targets a 7 días: Para predecir dentro de una semana, movemos la columna hacia arriba 7 posiciones.

        - Lluvia binaria: Convertimos la lluvia en 0/1 para poder hacer clasificación.

    - EXPLICACIÓN TÉCNICA (nivel intermedio)
        - Esta explicación es para alguien que ya entiende machine learning:

        - Lags: Introducen dependencia temporal explícita, permitiendo que el modelo aprenda patrones autoregresivos.

        - Rolling windows: Capturan tendencias de corto plazo y reducen ruido, mejorando la estabilidad del modelo.

        - Features temporales: Mes, día del año y semana introducen estacionalidad, esencial en series meteorológicas.

        - Interpolación: Los NaN generados por transformaciones temporales se imputan para evitar pérdida de información y mantener continuidad.

        - Targets desplazados: shift(-7) crea un problema supervisado donde las features actuales predicen valores futuros.

        - Lluvia binaria: Se transforma la precipitación en una variable categórica para modelos de clasificación.

    - EXPLICACIÓN PROFESIONAL (nivel experto)
        - Esta explicación es para un informe o presentación formal:

        - Ingeniería de características temporales:  
            La inclusión de lags y rolling windows permite modelar la estructura autoregresiva y las tendencias subyacentes de la serie, fundamentales en fenómenos meteorológicos donde la persistencia y la inercia climática son determinantes.

        - Codificación de estacionalidad:  
            La descomposición de la fecha en componentes cíclicos (mes, día del año, semana) introduce información estacional explícita, permitiendo que el modelo capture variaciones periódicas anuales y subanuales.

        - Imputación controlada:  
            La interpolación lineal se aplica únicamente a valores faltantes derivados de transformaciones matemáticas, preservando la integridad del dataset sin introducir sesgos significativos.

        - Formulación del problema supervisado:  
            El desplazamiento de las variables objetivo en un horizonte de 7 días define un problema de forecasting indirecto, donde las observaciones actuales se utilizan para anticipar condiciones futuras.

        - Clasificación binaria de precipitación:  
            La binarización de la lluvia facilita la construcción de modelos robustos de predicción de eventos, especialmente útil cuando la magnitud exacta de la precipitación es menos relevante que la ocurrencia del fenómeno.

    VERSIÓN COMBINADA:
        - En este proyecto he generado varias características nuevas para mejorar la capacidad predictiva del modelo.
        
        Primero, añadí lags porque en meteorología el estado del tiempo de días anteriores influye directamente en el futuro. También incluí rolling windows para capturar tendencias de varios días y reducir el ruido de valores aislados.
        
        A partir de la fecha extraje mes, día del año y semana, ya que el clima tiene una fuerte componente estacional y el modelo necesita esta información para aprender patrones anuales.
        
        Como estas transformaciones generan valores nulos, utilicé interpolación lineal para mantener la continuidad de la serie sin perder datos.
        
        Para convertir el problema en uno supervisado, desplacé las variables objetivo 7 días hacia adelante, de forma que las observaciones actuales sirven para predecir el clima de la semana siguiente.
        
        Finalmente, transformé la precipitación en una variable binaria para poder entrenar un modelo de clasificación que determine si lloverá o no.
        
        En conjunto, estas features permiten que el modelo capture tanto la dinámica temporal como la estacionalidad y las tendencias del clima, lo que mejora significativamente la calidad de las predicciones.

¿Por qué usar interpolación lineal y no polinómica u otro método? Porque es la opción más segura, estable y coherente para datos meteorológicos transformados (lags y rolling).

La interpolación lineal es la más coherente para datos que NO son reales, sino huecos matemáticos
Los NaN que aparecen en tu dataset no representan datos perdidos del mundo real, sino huecos creados por:

    shift() (lags)

    rolling() (ventanas móviles)

Es decir:

    No estás reconstruyendo datos meteorológicos reales, sino rellenando huecos artificiales.

Por eso:

    No necesitas un método complejo para “adivinar” nada.  
    Solo necesitas mantener la continuidad de la serie.

La interpolación lineal hace exactamente eso.

Ejemplo típico:

20°C, NaN, NaN, 22°C 

Interpolación lineal → 20, 20.66, 21.33, 22
Interpolación polinómica → 20, 19, 23, 22
Forward fill también es válido, pero menos suave → 20, 20, 20, 22

La temperatura, la humedad o la precipitación no cambian de forma polinómica entre dos días consecutivos.

El comportamiento real es:

    gradual

    suave

    continuo

La interpolación lineal respeta eso.

La temperatura, la humedad o la precipitación no cambian de forma polinómica entre dos días consecutivos.

El comportamiento real es:

    gradual

    suave

    continuo

La interpolación lineal respeta eso.

Para las transformaciones temporales (lags y rolling windows) se generan valores nulos que no corresponden a datos reales faltantes, sino a huecos matemáticos derivados del desplazamiento y las ventanas móviles.

Primero aplico interpolación lineal porque preserva la continuidad de la serie sin introducir oscilaciones artificiales. En caso de que alguna columna no pueda interpolarse correctamente, utilizo la media como método de imputación secundario, ya que mantiene la distribución general de los datos sin generar valores extremos.

Los valores nulos de las variables objetivo desplazadas (targets a 7 días) no se imputan, ya que representan información futura que no existe en el dataset original. Por ello, esas filas se eliminan.

Este enfoque garantiza un dataset consistente, continuo y adecuado para entrenar modelos de predicción meteorológica.