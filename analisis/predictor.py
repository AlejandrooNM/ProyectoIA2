import os
import cv2
import nibabel as nib
import numpy as np
import torch

# Rutas
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RUTA_MODELO = os.path.join(BASE_DIR, "modelos", "modelo_tumores.pth")

IMG_SIZE = 128

# Cargar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from modelos.modelo_unet import UNet
modelo = UNet().to(device)

if os.path.exists(RUTA_MODELO):
    modelo.load_state_dict(torch.load(RUTA_MODELO, map_location=device))
    modelo.eval()
    print(f"[GPU] Modelo cargado en: {device}")
else:
    print(f"Alerta: No se encontró el modelo en {RUTA_MODELO}")
    modelo = None

def predecir_tumor(ruta_imagen):
    img_obj  = nib.load(ruta_imagen)
    img_data = np.asanyarray(img_obj.dataobj)  # type: ignore

    # Slice central
    indice_medio = img_data.shape[2] // 2
    slice_central = img_data[:, :, indice_medio]

    # Corregir orientación NIfTI
    slice_central = slice_central.T
    original = slice_central.copy()

    # Preprocesamiento
    slice_resize = cv2.resize(slice_central, (IMG_SIZE, IMG_SIZE))
    max_val = np.max(slice_resize)
    if max_val > 0:
        slice_resize = slice_resize / max_val

    # Convertir a tensor PyTorch (1, 1, 128, 128)
    entrada = torch.tensor(slice_resize, dtype=torch.float32)
    entrada = entrada.unsqueeze(0).unsqueeze(0).to(device)

    # Predicción
    if modelo is not None:
        with torch.no_grad():
            prediccion = modelo(entrada)

        # Convertir salida a numpy (128, 128)
        mascara_pequeña = prediccion.squeeze().cpu().numpy()
        mascara_pequeña = (mascara_pequeña > 0.085).astype(np.uint8)

        # Redimensionar al tamaño real del original
        mascara = cv2.resize(
            mascara_pequeña,
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    else:
        mascara = np.zeros(original.shape, dtype=np.uint8)

    return original, mascara