import base64
import os
import mimetypes
from datetime import timedelta
import pandas as pd
import streamlit.components.v1 as components

"""tarjetas que giran"""

def img_to_base64(path):
    """Convierte imagen a string base64"""
    if not os.path.exists(path):
        return None
    try:
        mime, _ = mimetypes.guess_type(path)
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:{mime or 'image/png'};base64,{data}"
    except:
        return None

def obtener_icono_tiempo(lluvia, tmin, tmax, nubes):
    t_avg = (tmin + tmax) / 2
    if lluvia == 1:
        if t_avg < 2: return "nieve.png"
        if t_avg < 5: return "sol_y_nieve.png"
        return "sol_y_lluvia.png"
    if nubes > 75: return "nubes.png"
    if nubes > 30: return "sol_y_nube.png"
    return "sol.png"

def obtener_consejo(tmin, tmax, lluvia):
    # 1. Prioridad absoluta: Lluvia
    if lluvia == 1: 
        if tmin < 8:
            return "Abrigate y llevate un paraguas", "lleva_paraguas_abrigado.png"
        elif tmin < 18:
            return "llevate un paraguas y una chaqueta", "lleva_paraguas_fresco.png"
        elif tmax > 28:
            return "llevate un paraguas y cuidado con el calor", "lleva_paraguas_calor.png"
        else:
            return "llevate un paraguas", "lleva_paraguas_sin_frio.png"
    
    if lluvia == 0:
        #Frío por la mañana o por la tarde , horas extremas, sea alumno de mañana o de tarde. 
        if tmin < 8: 
            return " Hace frio, abrígate bien.", "abrigate.png"
        
        #Si va ha hacer fresco , igual mañana o tarde avisamos de la chaqueta.
        if tmin < 18:
            return "Día fresco. No olvides ponerte una chaqueta o sudadera.", "fresco.png"

        #Si va ha hacer Calor avisamos de hidratación
        if tmax > 28: 
            return "Hidratate y vete por la sombra", "botella_de_agua.png"
            
        #Buen tiempo
        return "Hace un dia genial", "dia_agradable.png"

def generar_grid_html(df, p_tmax, p_tmin, p_rain, base_path):
    """
    Genera y RENDERIZA el componente HTML aislado.
    Ya no devuelve un string, sino que pinta directamente el componente.
    """
    
    images_dir = os.path.join(base_path, 'images')
    cards_html = ""
    
    dias_es = {"Monday":"LUN", "Tuesday":"MAR", "Wednesday":"MIÉ", "Thursday":"JUE", "Friday":"VIE", "Saturday":"SÁB", "Sunday":"DOM"}
    fecha_hoy = pd.to_datetime("today")

    # Generamos el HTML interno de las cartas
    for i in range(len(df)):
        # Aseguramos límite de 7 días para que la fila sea perfecta
        if i >= 7: break 

        fecha = fecha_hoy + timedelta(days=i)
        dia_nom = "HOY" if i == 0 else dias_es.get(fecha.strftime('%A'), "DÍA")
        fecha_fmt = fecha.strftime('%d/%m')
        
        val_tmax = p_tmax[i]
        val_tmin = p_tmin[i]
        val_rain = p_rain[i]
        val_nubes = df.iloc[i]['cloudcover__mean'] if 'cloudcover__mean' in df.columns else 0

        icon_name = obtener_icono_tiempo(val_rain, val_tmin, val_tmax, val_nubes)
        txt_tip, icon_tip_name = obtener_consejo(val_tmin, val_tmax, val_rain)
        
        b64_main = img_to_base64(os.path.join(images_dir, icon_name))
        b64_tip = img_to_base64(os.path.join(images_dir, icon_tip_name))
        
        img_main = f'<img src="{b64_main}" class="main-icon">' if b64_main else '<div style="font-size:40px"></div>'
        img_tip = f'<img src="{b64_tip}" class="tip-icon">' if b64_tip else '<div style="font-size:30px"></div>'
        badge = '<div class="badge-rain">Lluvia</div>' if val_rain == 1 else ''

        cards_html += f"""
        <div class="card">
            <div class="card-inner">
                <div class="card-front">
                    <div class="day">{dia_nom}</div>
                    <div class="date">{fecha_fmt}</div>
                    {img_main}
                    <div class="temp-box">
                        <span class="tmax">{val_tmax:.0f}°</span>
                        <span class="tmin">{val_tmin:.0f}°</span>
                    </div>
                    {badge}
                </div>

                <div class="card-back" style="background-image: url('{b64_tip}');">
                    <div class="overlay-consejo">
                        <div class="tip-title">RECOMENDACIÓN</div>
                        <div class="tip-spacer"></div> <div class="tip-text">{txt_tip}</div>
                    </div>
                </div>
            </div>
        </div>
        """

    # HTML COMPLETO CON CSS INCRUSTADO (AISLADO DE STREAMLIT)
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700;900&display=swap');
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Roboto', sans-serif; }}
        
        body {{ 
            background-color: transparent; 
            overflow: hidden; /* Sin scroll vertical global */
        }}

        /* GRID PRINCIPAL: 7 COLUMNAS */
        .grid-container {{
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 10px;
            width: 100%;
            padding: 5px;
        }}

        /* CARTA BASE */
        .card {{
            background-color: transparent;
            height: 240px; /* Altura fija */
            perspective: 1000px;
            cursor: pointer;
        }}

        .card-inner {{
            position: relative;
            width: 100%;
            height: 100%;
            text-align: center;
            transition: transform 0.6s cubic-bezier(0.4, 0.2, 0.2, 1);
            transform-style: preserve-3d;
            border-radius: 16px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }}

        .card:hover .card-inner {{ transform: rotateY(180deg); }}

        .card-front, .card-back {{
            position: absolute;
            width: 100%;
            height: 100%;
            -webkit-backface-visibility: hidden;
            backface-visibility: hidden;
            border-radius: 16px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 10px;
        }}

        /* DISEÑO FRONTAL */
        .card-front {{
            background: linear-gradient(160deg, #1e293b, #0f172a);
            border: 1px solid #334155;
            color: #fff;
        }}

        .day {{ font-weight: 900; font-size: 16px; color: #cbd5e1; letter-spacing: 1px; margin-bottom: 2px; }}
        .date {{ font-size: 12px; color: #64748b; margin-bottom: 10px; }}
        .main-icon {{ width: 60px; height: 60px; margin-bottom: 5px; object-fit: contain; }}
        
        .temp-box {{ margin: 5px 0; font-size: 14px; }}
        .tmax {{ font-size: 24px; font-weight: 800; color: #fff; display: block; }}
        .tmin {{ font-size: 14px; color: #94a3b8; }}

        .badge-rain {{
            background: rgba(59, 130, 246, 0.2);
            color: #60a5fa;
            font-size: 10px;
            padding: 3px 8px;
            border-radius: 10px;
            margin-top: 5px;
            font-weight: bold;
        }}

        /* DISEÑO TRASERO */

        .card-back {{
            position: relative;
            transform: rotateY(180deg);
            border-radius: 16px;
            background-size: cover;
            background-position: center;
            overflow: hidden;
            border: 1px solid #60a5fa;
        }}

        .overlay-consejo {{
            background: transparent; 
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between; 
            padding: 20px 15px;
            text-align: center
        }}

        .tip-title {{ 
            font-size: 11px;
            font-weight: 900;
            color: #ffffff; /* Puedes ponerlo blanco o un color que resalte */
            letter-spacing: 2px;
            text-shadow: 1px 1px 3px rgba(0,0,0,1), 0px 0px 8px rgba(0,0,0,0.8);
        }}

        .tip-spacer {{
            flex-grow: 1; 
        }}

        .tip-text {{ 
            font-size: 15px; 
            font-weight: 800; 
            color: #ffffff;
            line-height: 1.2;
            background: transparent; 
            padding: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,1),   /* Sombra sólida desplazada */
                         0px 0px 10px rgba(0,0,0,0.9), /* Brillo oscuro suave alrededor */
                         0px 0px 20px rgba(0,0,0,0.8); /* Difuminado extra para profundidad */
        }}

        /* RESPONSIVE: Scroll horizontal en móvil */
        @media (max-width: 800px) {{
            .grid-container {{
                display: flex;
                overflow-x: auto;
                padding-bottom: 15px;
            }}
            .card {{
                min-width: 110px; /* Ancho mínimo en móvil */
            }}
        }}
    </style>
    </head>
    <body>
        <div class="grid-container">
            {cards_html}
        </div>
    </body>
    </html>
    """
    
    # Renderizamos el componente con una altura segura para que no corte la sombra
    components.html(full_html, height=270, scrolling=False)