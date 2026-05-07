import os
import cv2
import nibabel as nib
import numpy as np

IMG_SIZE = 128

def cargar_datos(dataset_path):
    imagenes = []
    mascaras = []

    # Verificamos que la ruta exista para evitar errores
    if not os.path.exists(dataset_path):
        print(f"Error: No se encuentra la carpeta {dataset_path}")
        return np.array([]), np.array([])

    pacientes = os.listdir(dataset_path)

    for paciente in pacientes:
        ruta_paciente = os.path.join(dataset_path, paciente)
        if not os.path.isdir(ruta_paciente):
            continue

        archivos = os.listdir(ruta_paciente)
        flair = None
        seg = None

        for archivo in archivos:
            if "flair" in archivo.lower():
                flair = os.path.join(ruta_paciente, archivo)
            if "seg" in archivo.lower():
                seg = os.path.join(ruta_paciente, archivo)

        if flair and seg:
            print(f"Procesando: {paciente}...")
            # Usamos type: ignore para que VS Code no marque error donde no lo hay
            img_obj = nib.load(flair)
            mask_obj = nib.load(seg)
            
            img = img_obj.get_fdata() # type: ignore
            mask = mask_obj.get_fdata() # type: ignore

            total_slices = img.shape[2]

            for i in range(total_slices):
                imagen_slice = img[:, :, i]
                mask_slice = mask[:, :, i]

                # Solo guardamos rebanadas que contengan tumor para no saturar la memoria
                # y para que la IA aprenda mejor qué es un tumor
                if np.max(mask_slice) > 0:
                    imagen_slice = cv2.resize(imagen_slice, (IMG_SIZE, IMG_SIZE))
                    mask_slice = cv2.resize(mask_slice, (IMG_SIZE, IMG_SIZE))

                    # Normalización segura
                    max_img = np.max(imagen_slice)
                    if max_img > 0:
                        imagen_slice = imagen_slice / max_img

                    mask_slice = (mask_slice > 0).astype(np.float32)

                    imagenes.append(imagen_slice)
                    mascaras.append(mask_slice)

    # Convertir a arrays de numpy y darles la forma (Samples, Ancho, Alto, Canal)
    imagenes_final = np.array(imagenes).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    mascaras_final = np.array(mascaras).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    return imagenes_final, mascaras_final