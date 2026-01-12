import os

def obtener_icono_tiempo(lluvia_bin, t_min, t_max, nubes_mean):
    """
    Retorna nombre de imagen de icono del clima según lluvia, temperatura y nubosidad.
    Parámetros: lluvia_bin (int 0/1), t_min (float), t_max (float), nubes_mean (float).
    Retorna: str nombre archivo PNG.
    """
    t_promedio = (t_min + t_max) / 2

    # 1. Lluvia y Nieve
    if lluvia_bin == 1:
        if t_promedio < 2: 
            return "nieve.png"  # Hace mucho frío y precipita
        elif t_promedio < 5:
            return "sol_y_nieve.png" # Agua nieve o frío extremo
        else:
            return "sol_y_lluvia.png" # Lluvia estándar

    # 2. Nubosidad (sin lluvia)
    if nubes_mean > 75:
        return "nubes.png"
    elif nubes_mean > 30:
        return "sol_y_nube.png"
    
    # 3. Sol
    return "sol.png"


def obtener_consejo_y_kit(t_min, t_max, lluvia_bin):
    """
    Genera recomendación de qué llevar según lluvia y temperatura predichas.
    Parámetros: t_min (float), t_max (float), lluvia_bin (int 0/1).
    Retorna: tupla (str consejo, str nombre_icono).
    """
    t_promedio = (t_min + t_max) / 2
    
    # Prioridad 1: Lluvia
    if lluvia_bin == 1:
        return "¡Va a llover! No olvides el paraguas.", "paraguas.png"
    
    # Prioridad 2: Temperatura Extrema
    if t_promedio < 12:
        return "Hace frío. Abrígate bien.", "chaqueta.png"
    elif t_promedio > 25:
        return "Mucho calor. ¡Hidrátate!", "botella_de_agua.png"
    
    # Prioridad 3: Tiempo estándar (usamos chaqueta ligera o nada, por defecto chaqueta si refresca)
    if t_promedio < 20:
        return "Refresca un poco, lleva una capa extra.", "chaqueta.png"
    else:
        return "Tiempo agradable. Disfruta del día.", "sol.png" # Si no hace nada especial, sol
    