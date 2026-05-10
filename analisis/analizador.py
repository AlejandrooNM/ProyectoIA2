import numpy as np

def obtener_diagnostico_clinico(mascara_2d):
    # Escala BraTS: 1px approx 1.87mm
    ESCALA_PIXEL_MM = 1.87
    area_px = np.sum(mascara_2d > 0.5)
    area_cm2 = (area_px * (ESCALA_PIXEL_MM**2)) / 100

    volumen_cm3 = (area_cm2 ** 1.5) * 0.75
    diametro_cm = np.sqrt(area_cm2 * 4 / np.pi)

    if diametro_cm < 2.0:
        riesgo = "BAJO"
    elif diametro_cm <= 5.0:
        riesgo = "MODERADO"
    else:
        riesgo = "ALTO (CRITICO)"

    centro_x = mascara_2d.shape[1] // 2

    # FIX: En neuroimagen radiológica la izquierda del paciente
    # aparece a la derecha en pantalla, por eso invertimos la lógica
    lado = "IZQUIERDO" if np.sum(mascara_2d[:, centro_x:]) > np.sum(mascara_2d[:, :centro_x]) else "DERECHO"

    return {
        "volumen": f"{volumen_cm3:.2f} cm3",
        "ubicacion": f"HEMISFERIO {lado}" if area_px > 0 else "N/A",
        "riesgo": riesgo,
        "diametro": f"{diametro_cm:.2f} cm"
    }