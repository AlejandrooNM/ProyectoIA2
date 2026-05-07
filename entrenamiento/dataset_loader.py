import os
import nibabel as nib
import numpy as np
import cv2

IMG_SIZE = 128

def cargar_datos(ruta_dataset, num_pacientes=50):
    imagenes = []
    mascaras = []
    
    # Obtener lista de carpetas de pacientes
    pacientes = [f for f in os.listdir(ruta_dataset) if os.path.isdir(os.path.join(ruta_dataset, f))]
    pacientes = pacientes[:num_pacientes]
    
    print(f"Buscando datos en: {ruta_dataset}")

    for p in pacientes:
        ruta_p = os.path.join(ruta_dataset, p)
        archivos = os.listdir(ruta_p)
        
        t1ce_path = None
        seg_path = None
        
        # Ajustamos para que busque tus archivos .nii
        for arc in archivos:
            if "_t1ce.nii" in arc.lower():
                t1ce_path = os.path.join(ruta_p, arc)
            elif "_seg.nii" in arc.lower():
                seg_path = os.path.join(ruta_p, arc)
        
        if t1ce_path and seg_path:
            try:
                # El # type: ignore quita los errores rojos de VS Code
                img_obj = nib.load(t1ce_path)
                mask_obj = nib.load(seg_path)
                
                img_data = img_obj.get_fdata() # type: ignore
                mask_data = mask_obj.get_fdata() # type: ignore
                
                # Recorrer las rebanadas del volumen 3D
                for i in range(img_data.shape[2]):
                    mask_slice = mask_data[:, :, i]
                    
                    # Filtro de tumor (más de 10 píxeles activos)
                    if np.sum(mask_slice > 0) > 10:
                        img_slice = img_data[:, :, i]
                        
                        img_res = cv2.resize(img_slice, (IMG_SIZE, IMG_SIZE))
                        mask_res = cv2.resize(mask_slice, (IMG_SIZE, IMG_SIZE))
                        
                        # Normalización
                        if img_res.max() > img_res.min():
                            img_res = (img_res - img_res.min()) / (img_res.max() - img_res.min())
                        
                        mask_res = (mask_res > 0).astype(np.float32)
                        
                        imagenes.append(img_res)
                        mascaras.append(mask_res)
                
                print(f"✅ Paciente {p} procesado con éxito.")
            except Exception as e:
                print(f" Error en paciente {p}: {e}")
                
    return np.array(imagenes).reshape(-1, IMG_SIZE, IMG_SIZE, 1), \
           np.array(mascaras).reshape(-1, IMG_SIZE, IMG_SIZE, 1)