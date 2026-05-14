import os
import cv2
import nibabel as nib
import numpy as np
import torch

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RUTA_MODELO = os.path.join(BASE_DIR, "modelos", "modelo_tumores.pth")
IMG_SIZE    = 128

# MEJORA 1 — Umbral estándar (era 0.085, demasiado permisivo → muchos falsos positivos)
UMBRAL_PREDICCION = 0.5

# MEJORA 2 — Dispositivo global; el modelo se carga bajo demanda (lazy loading)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_modelo_cache = None   # caché: se inicializa solo la primera vez que se llama


def _get_modelo():
    """
    Carga la U-Net en memoria la primera vez que se invoca y la reutiliza
    en llamadas posteriores. Evita recargar el modelo en cada importación
    del módulo (problema original).
    """
    global _modelo_cache
    if _modelo_cache is not None:
        return _modelo_cache

    from modelos.modelo_unet import UNet
    modelo = UNet().to(device)

    if not os.path.exists(RUTA_MODELO):
        print(f"[ALERTA] Modelo no encontrado en: {RUTA_MODELO}")
        return None

    modelo.load_state_dict(torch.load(RUTA_MODELO, map_location=device))
    modelo.eval()
    print(f"[OK] Modelo cargado en: {device}")
    _modelo_cache = modelo
    return _modelo_cache


def _preprocesar_slice(slice_2d):
    """Redimensiona y normaliza un slice 2D listo para inferencia."""
    slice_res = cv2.resize(slice_2d.astype(np.float32), (IMG_SIZE, IMG_SIZE))
    max_val = slice_res.max()
    if max_val > 0:
        slice_res = slice_res / max_val
    tensor = torch.tensor(slice_res, dtype=torch.float32)
    return tensor.unsqueeze(0).unsqueeze(0).to(device)   # (1, 1, H, W)


def _inferir_slice(modelo, tensor_slice):
    """Devuelve la máscara de probabilidades (numpy, 128×128)."""
    with torch.no_grad():
        salida = modelo(tensor_slice)
    return salida.squeeze().cpu().numpy()   # (128, 128) valores en [0, 1]


def predecir_tumor(ruta_imagen):
    """
    Carga el volumen NIfTI, busca el slice con mayor activación tumoral,
    aplica la U-Net y devuelve la imagen original del mejor slice junto
    con la máscara binaria, la máscara de probabilidades y las máscaras
    de todos los slices con tumor (para cálculo de volumen 3D).

    Returns:
        original       : np.ndarray (H, W) — slice axial en escala original
        mascara        : np.ndarray (H, W) — máscara binaria redimensionada
        mascara_prob   : np.ndarray (H, W) — probabilidades crudas de la red
        slices_mascara : list[np.ndarray]  — máscaras binarias de todos los
                         slices donde se detectó tumor (para volumen 3D)
    """

    # ── MEJORA 3 — Validación de entrada ──────────────────────────────────────
    if not os.path.isfile(ruta_imagen):
        raise FileNotFoundError(f"Archivo no encontrado: {ruta_imagen}")
    if not ruta_imagen.lower().endswith(('.nii', '.nii.gz')):
        raise ValueError(f"Formato no soportado. Se esperaba .nii o .nii.gz: {ruta_imagen}")

    img_obj  = nib.load(ruta_imagen)
    img_data = np.asanyarray(img_obj.dataobj)  # type: ignore

    if img_data.ndim != 3:
        raise ValueError(f"Se esperaba un volumen 3D, se recibió shape: {img_data.shape}")

    modelo = _get_modelo()

    # ── MEJORA 4 — Buscar el slice con mayor activación en lugar del central ──
    # Itera por todos los slices axiales y se queda con el que tenga mayor
    # suma de probabilidades predichas (= slice más representativo del tumor).
    mejor_indice = img_data.shape[2] // 2   # fallback al central si no hay modelo
    mejor_score  = -1.0
    scores_por_slice = []

    if modelo is not None:
        for i in range(img_data.shape[2]):
            s = img_data[:, :, i].T
            t = _preprocesar_slice(s)
            prob = _inferir_slice(modelo, t)
            score = float(prob.sum())
            scores_por_slice.append((i, score, prob))
            if score > mejor_score:
                mejor_score  = score
                mejor_indice = i

    # Slice original (sin redimensionar) del mejor índice
    slice_original = img_data[:, :, mejor_indice].T
    original = slice_original.copy()

    if modelo is None:
        mascara_vacia = np.zeros(original.shape, dtype=np.uint8)
        return original, mascara_vacia, mascara_vacia.astype(np.float32), []

    # Máscara del mejor slice
    _, _, prob_mejor = scores_por_slice[mejor_indice]
    mascara_prob_128  = prob_mejor                                       # (128, 128)
    mascara_bin_128   = (mascara_prob_128 > UMBRAL_PREDICCION).astype(np.uint8)

    # Redimensionar al tamaño real del slice original
    h_orig, w_orig = original.shape
    mascara = cv2.resize(
        mascara_bin_128,
        (w_orig, h_orig),
        interpolation=cv2.INTER_NEAREST
    )
    mascara_prob_orig = cv2.resize(
        mascara_prob_128,
        (w_orig, h_orig),
        interpolation=cv2.INTER_LINEAR
    )

    # ── MEJORA 5 — Recopilar máscaras de todos los slices con tumor ───────────
    # Se usan en analizador.py para calcular el volumen real 3D.
    slices_mascara = []
    for i, score, prob in scores_por_slice:
        bin_slice = (prob > UMBRAL_PREDICCION).astype(np.uint8)
        if bin_slice.sum() >= 50:   # al menos 50 px activos
            slices_mascara.append(bin_slice)

    return original, mascara, mascara_prob_orig, slices_mascara