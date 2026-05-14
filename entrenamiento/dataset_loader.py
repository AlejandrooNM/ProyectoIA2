import os
import nibabel as nib
import numpy as np
import cv2
import random

IMG_SIZE        = 128
MIN_PIXELS_MASK = 100   # mínimo de píxeles positivos para incluir un slice

# ── Augmentación ──────────────────────────────────────────────────────────────

def aumentar(img, mask):
    """
    Aplica aumentaciones aleatorias simétricas sobre imagen y máscara.
    Todas las transformaciones son reversibles o invariantes a la semántica
    médica del slice axial:
      - Volteo horizontal  (lateralidad se conserva por convención)
      - Rotación pequeña   (±15°)
      - Ajuste de brillo   (multiplicación ×[0.85, 1.15])
    """
    # Volteo horizontal
    if random.random() < 0.5:
        img  = cv2.flip(img,  1)
        mask = cv2.flip(mask, 1)

    # Rotación aleatoria ±15 grados
    if random.random() < 0.5:
        angulo = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), angulo, 1.0)
        img  = cv2.warpAffine(img,  M, (IMG_SIZE, IMG_SIZE),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, M, (IMG_SIZE, IMG_SIZE),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_REFLECT_101)

    # Ajuste de brillo (solo sobre la imagen, no la máscara)
    if random.random() < 0.5:
        factor = random.uniform(0.85, 1.15)
        img    = np.clip(img * factor, 0.0, 1.0)

    return img.astype(np.float32), mask.astype(np.float32)


# ── Procesamiento por paciente ────────────────────────────────────────────────

def procesar_un_paciente(ruta_p, augmentar=False):
    """
    Carga los archivos t1ce y seg de un paciente y devuelve los slices válidos.

    Args:
        ruta_p    : carpeta del paciente (contiene los .nii)
        augmentar : si True, aplica aumentación a cada slice antes de devolverlo

    Returns:
        (imagenes, mascaras) — listas de arrays (IMG_SIZE, IMG_SIZE)
    """
    imagenes_paciente = []
    mascaras_paciente = []

    archivos   = os.listdir(ruta_p)
    t1ce_path  = None
    seg_path   = None

    for arc in archivos:
        arc_lower = arc.lower()
        if "_t1ce.nii" in arc_lower:
            t1ce_path = os.path.join(ruta_p, arc)
        elif "_seg.nii" in arc_lower:
            seg_path  = os.path.join(ruta_p, arc)

    if not (t1ce_path and seg_path):
        return imagenes_paciente, mascaras_paciente

    img_data  = np.asanyarray(nib.load(t1ce_path).dataobj)   # type: ignore
    mask_data = np.asanyarray(nib.load(seg_path).dataobj)    # type: ignore

    for i in range(img_data.shape[2]):
        mask_slice = mask_data[:, :, i]

        # Filtro: descartar slices casi vacíos
        if np.sum(mask_slice > 0) < MIN_PIXELS_MASK:
            continue

        img_slice  = img_data[:, :, i]

        # Corregir orientación NIfTI → convención matplotlib / radiológica
        img_slice  = img_slice.T
        mask_slice = mask_slice.T

        # Redimensionamiento
        img_res  = cv2.resize(img_slice.astype(np.float32),
                              (IMG_SIZE, IMG_SIZE),
                              interpolation=cv2.INTER_LINEAR)
        mask_res = cv2.resize(mask_slice.astype(np.float32),
                              (IMG_SIZE, IMG_SIZE),
                              interpolation=cv2.INTER_NEAREST)  # NEAREST para máscaras

        # MEJORA — Normalización Z-score por slice en lugar de Min-Max global
        # Es más robusta ante artefactos de imagen con intensidades extremas.
        mean = img_res.mean()
        std  = img_res.std()
        if std > 1e-6:
            img_res = (img_res - mean) / std
            # Clipear a ±3σ y re-escalar a [0, 1]
            img_res = np.clip(img_res, -3.0, 3.0)
            img_res = (img_res + 3.0) / 6.0
        else:
            img_res = np.zeros_like(img_res)

        # Binarización de la máscara (etiquetas BraTS: 1, 2, 4 → todo es tumor)
        mask_res = (mask_res > 0).astype(np.float32)

        # Aumentación (solo en entrenamiento)
        if augmentar:
            img_res, mask_res = aumentar(img_res, mask_res)

        imagenes_paciente.append(img_res)
        mascaras_paciente.append(mask_res)

    return imagenes_paciente, mascaras_paciente


# ── Cargador principal ────────────────────────────────────────────────────────

def cargar_datos(lista_rutas_pacientes, augmentar=False):
    """
    Carga y preprocesa todos los pacientes de la lista.

    Args:
        lista_rutas_pacientes : list[str]
        augmentar             : bool — activar augmentación (usar True solo en train)

    Returns:
        X_array : np.ndarray (N, IMG_SIZE, IMG_SIZE, 1)
        y_array : np.ndarray (N, IMG_SIZE, IMG_SIZE, 1)
    """
    X = []
    y = []

    for ruta in lista_rutas_pacientes:
        try:
            imgs, masks = procesar_un_paciente(ruta, augmentar=augmentar)
            X.extend(imgs)
            y.extend(masks)
        except Exception as e:
            print(f"[ERROR] Paciente omitido ({ruta}): {e}")

    X_array = np.array(X, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_array = np.array(y, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    return X_array, y_array