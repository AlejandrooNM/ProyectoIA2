import os
import cv2
import nibabel as nib
import numpy as np
import keras

IMG_SIZE = 128

# Mantenemos la lógica de rutas dinámicas para que no falle en Windows
RUTA_MODELO = os.path.join("modelos", "modelo_tumores.keras")

if os.path.exists(RUTA_MODELO):
    modelo = keras.models.load_model(RUTA_MODELO)
else:
    print(f"Alerta: No se encontró el modelo en {RUTA_MODELO}")
    modelo = None

def predecir_tumor(ruta_imagen):
    # 1. Cargar archivo NIfTI de forma segura para el editor
    # Al separar el objeto de carga del método get_fdata, Pylance suele dejar de marcar error
    img_obj = nib.load(ruta_imagen)
    # 1. Cargar archivo NIfTI
    img_obj = nib.load(ruta_imagen)
    # Añadimos "# type: ignore" al final para que VS Code deje de marcar error
    img_data = np.asanyarray(img_obj.dataobj) # type: ignore

    # 2. Extraer el corte central (Slice)
    # Las MRI son volúmenes 3D (X, Y, Z). Tomamos el medio del eje Z.
    indice_medio = img_data.shape[2] // 2
    slice_central = img_data[:, :, indice_medio]

    # 3. Preprocesamiento
    original = slice_central.copy()
    
    # Redimensionar al tamaño que espera la U-Net (128x128)
    slice_resize = cv2.resize(slice_central, (IMG_SIZE, IMG_SIZE))
    
    # Normalización: los valores de MRI pueden ser muy altos, los llevamos de 0 a 1
    max_val = np.max(slice_resize)
    if max_val > 0:
        slice_resize = slice_resize / max_val
    
    # Ajustar dimensiones para Keras: (1, 128, 128, 1)
    entrada = slice_resize.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # 4. Predicción con el modelo cargado
    if modelo is not None:
        prediccion = modelo.predict(entrada, verbose=0)[0]
        # Binarización (Umbral 0.5)
        mascara = (prediccion > 0.5).astype(np.uint8)
    else:
        mascara = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    # Retornamos la imagen original para mostrarla y la máscara para el reporte
    return original, mascara