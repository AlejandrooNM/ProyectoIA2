import os
import nibabel as nib
import numpy as np
import cv2

IMG_SIZE = 128

def procesar_un_paciente(ruta_p):
    imagenes_paciente = []
    mascaras_paciente = []

    archivos = os.listdir(ruta_p)
    t1ce_path = None
    seg_path = None

    for arc in archivos:
        if "_t1ce.nii" in arc.lower():
            t1ce_path = os.path.join(ruta_p, arc)
        elif "_seg.nii" in arc.lower():
            seg_path = os.path.join(ruta_p, arc)

    if t1ce_path and seg_path:
        img_data = np.asanyarray(nib.load(t1ce_path).dataobj)  # type: ignore
        mask_data = np.asanyarray(nib.load(seg_path).dataobj)  # type: ignore

        for i in range(img_data.shape[2]):
            mask_slice = mask_data[:, :, i]

            # FIX 1: Subimos filtro de 10 a 100 píxeles para evitar slices casi vacíos
            if np.sum(mask_slice > 0) < 100:
                continue

            img_slice = img_data[:, :, i]

            # FIX 2: Transpose para corregir orientación NIfTI -> matplotlib
            img_slice  = img_slice.T
            mask_slice = mask_slice.T

            # Redimensionamiento
            img_res  = cv2.resize(img_slice,  (IMG_SIZE, IMG_SIZE))
            mask_res = cv2.resize(mask_slice, (IMG_SIZE, IMG_SIZE),
                                  interpolation=cv2.INTER_NEAREST)  # FIX 3: NEAREST para máscaras binarias

            # Normalización Min-Max
            if img_res.max() > img_res.min():
                img_res = (img_res - img_res.min()) / (img_res.max() - img_res.min())

            # Binarización de la máscara
            mask_res = (mask_res > 0).astype(np.float32)

            imagenes_paciente.append(img_res)
            mascaras_paciente.append(mask_res)

    return imagenes_paciente, mascaras_paciente


def cargar_datos(lista_rutas_pacientes):
    X = []
    y = []

    for ruta in lista_rutas_pacientes:
        try:
            imgs, masks = procesar_un_paciente(ruta)
            X.extend(imgs)
            y.extend(masks)
        except Exception as e:
            print(f"Error cargando paciente {ruta}: {e}")

    X_array = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_array = np.array(y).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    return X_array, y_array