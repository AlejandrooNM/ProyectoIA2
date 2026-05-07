import numpy as np

def analizar_tumor(mascara):
    # Aseguramos que la máscara sea 2D (por si viene como 128, 128, 1)
    if mascara.ndim == 3:
        mascara = np.squeeze(mascara)

    # Contar píxeles que la IA marcó como tumor (valores cercanos a 1)
    # Usamos un umbral de 0.5 para binarizar la salida de la sigmoide
    pixeles_tumor = np.sum(mascara > 0.5)

    # Calcular porcentaje sobre el área total (128x128 = 16384 píxeles)
    porcentaje = (pixeles_tumor / (128 * 128)) * 100

    # Clasificación de riesgo
    if porcentaje == 0:
        riesgo = "Nulo"
    elif porcentaje < 5:
        riesgo = "Bajo"
    elif porcentaje < 15:
        riesgo = "Moderado"
    else:
        riesgo = "Alto"

    ubicacion = detectar_ubicacion(mascara)

    return {
        "pixeles_tumor": int(pixeles_tumor),
        "porcentaje": round(porcentaje, 2),
        "riesgo": riesgo,
        "ubicacion": ubicacion
    }

def detectar_ubicacion(mascara):
    # Obtenemos las coordenadas de los píxeles del tumor (> 0.5)
    coords = np.argwhere(mascara > 0.5)

    if len(coords) == 0:
        return "No detectado"

    # Coordenada X promedio (Eje horizontal)
    # coords[:, 1] son las columnas (X)
    promedio_x = np.mean(coords[:, 1])

    # Coordenada Y promedio (Eje vertical) para saber si es frontal o posterior
    promedio_y = np.mean(coords[:, 0])

    # Lógica de ubicación mejorada
    lado = "Izquierdo" if promedio_x < 64 else "Derecho"
    zona = "Frontal" if promedio_y < 64 else "Posterior"

    return f"Hemisferio {lado} - Zona {zona}"