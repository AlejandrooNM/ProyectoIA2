import numpy as np

# ── Constantes clínicas ───────────────────────────────────────────────────────
ESCALA_PIXEL_MM  = 1.87   # BraTS: 1 px ≈ 1.87 mm en plano axial
GROSOR_SLICE_MM  = 1.0    # BraTS usa slices de 1 mm de grosor
MIN_PIXELS_TUMOR = 50     # umbral mínimo para considerar tumor presente

def obtener_diagnostico_clinico(mascara_2d, mascara_prob=None, slices_mascara=None):
    """
    Calcula métricas clínicas a partir de la máscara de segmentación.

    Args:
        mascara_2d      : np.ndarray (H, W) — máscara binaria del mejor slice.
        mascara_prob    : np.ndarray (H, W) opcional — salida continua de la red
                          (probabilidades). Se usa para calcular el score de confianza.
        slices_mascara  : list[np.ndarray] opcional — lista de máscaras binarias de
                          todos los slices con tumor. Permite calcular volumen real 3D.

    Returns:
        dict con claves: area, diametro, volumen, ubicacion, riesgo, confianza,
                         slices_con_tumor, tumor_presente.
    """

    area_px = int(np.sum(mascara_2d > 0.5))

    # ── Sin tumor detectado ───────────────────────────────────────────────────
    if area_px < MIN_PIXELS_TUMOR:
        return {
            "tumor_presente" : False,
            "area"           : "0.00 cm2",
            "diametro"       : "0.00 cm",
            "volumen"        : "0.00 cm3",
            "ubicacion"      : "N/A",
            "riesgo"         : "SIN TUMOR DETECTADO",
            "confianza"      : "0.00 %",
            "slices_con_tumor": 0,
        }

    # ── Área 2D (mejor slice) ─────────────────────────────────────────────────
    area_cm2 = (area_px * (ESCALA_PIXEL_MM ** 2)) / 100.0

    # ── Volumen ───────────────────────────────────────────────────────────────
    # Si se proporcionan todos los slices, calculamos el volumen real sumando
    # la contribución de cada corte (área × grosor). Es mucho más preciso que
    # la heurística geométrica anterior (area^1.5 × 0.75).
    if slices_mascara is not None and len(slices_mascara) > 0:
        area_total_px = sum(int(np.sum(m > 0.5)) for m in slices_mascara)
        volumen_cm3 = (area_total_px * (ESCALA_PIXEL_MM ** 2) * GROSOR_SLICE_MM) / 1000.0
        num_slices = len(slices_mascara)
    else:
        # Fallback: heurística esférica (mantiene compatibilidad si no hay slices)
        volumen_cm3 = (area_cm2 ** 1.5) * 0.75
        num_slices = 1

    # ── Diámetro estimado (círculo equivalente) ───────────────────────────────
    diametro_cm = np.sqrt(area_cm2 * 4.0 / np.pi)

    # ── Clasificación de riesgo (criterios neurorradiológicos estándar) ───────
    if diametro_cm < 2.0:
        riesgo = "BAJO"
    elif diametro_cm <= 5.0:
        riesgo = "MODERADO"
    else:
        riesgo = "ALTO (CRITICO)"

    # ── Localización hemisférica ──────────────────────────────────────────────
    # CONVENCIÓN RADIOLÓGICA: izquierda de la imagen = hemisferio derecho del paciente
    centro_x = mascara_2d.shape[1] // 2
    suma_der_imagen = np.sum(mascara_2d[:, centro_x:])
    suma_izq_imagen = np.sum(mascara_2d[:, :centro_x])

    if suma_der_imagen > suma_izq_imagen:
        lado = "IZQUIERDO"   # lado imagen-derecha → paciente-izquierdo
    elif suma_izq_imagen > suma_der_imagen:
        lado = "DERECHO"     # lado imagen-izquierda → paciente-derecho
    else:
        lado = "CENTRAL"

    # ── Score de confianza ────────────────────────────────────────────────────
    # Media de las probabilidades crudas sobre la región positiva.
    # Refleja qué tan segura está la red de los píxeles que marcó como tumor.
    if mascara_prob is not None:
        pixeles_positivos = mascara_prob[mascara_2d > 0.5]
        confianza = float(np.mean(pixeles_positivos)) * 100.0 if len(pixeles_positivos) > 0 else 0.0
    else:
        confianza = 0.0

    return {
        "tumor_presente"  : True,
        "area"            : f"{area_cm2:.2f} cm2",
        "diametro"        : f"{diametro_cm:.2f} cm",
        "volumen"         : f"{volumen_cm3:.2f} cm3",
        "ubicacion"       : f"HEMISFERIO {lado}",
        "riesgo"          : riesgo,
        "confianza"       : f"{confianza:.1f} %",
        "slices_con_tumor": num_slices,
    }