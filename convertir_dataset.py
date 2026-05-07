import dicom2nifti
import os
import sys

# CONFIGURACIÓN DE RUTAS
# Asegúrate de que esta ruta sea la carpeta que contiene las subcarpetas 00000, 00001, etc.
ruta_dicom_raiz = r"E:\BraTS2021_TrainingSet_dcm\RSNA-ASNR-MICCAI-BraTS-2021\BraTS2021_TrainingSet_dcm" 
ruta_nifti_destino = r"E:\ProyectoIA2\dataset"

def convertir_training_set(num_pacientes=10):
    if not os.path.exists(ruta_nifti_destino):
        os.makedirs(ruta_nifti_destino)
        print(f"Carpeta creada: {ruta_nifti_destino}")

    try:
        pacientes = os.listdir(ruta_dicom_raiz)[:num_pacientes]
    except FileNotFoundError:
        print(f"Error: No se encontró la ruta {ruta_dicom_raiz}. Verifica que tu disco E: esté conectado.")
        return

    for paciente in pacientes:
        ruta_paciente_dicom = os.path.join(ruta_dicom_raiz, paciente)
        ruta_paciente_nifti = os.path.join(ruta_nifti_destino, paciente)

        if not os.path.isdir(ruta_paciente_dicom):
            continue

        if not os.path.exists(ruta_paciente_nifti):
            os.makedirs(ruta_paciente_nifti)

        print(f"\n--- Convirtiendo paciente: {paciente} ---")
        
        subcarpetas = os.listdir(ruta_paciente_dicom)
        
        for sub in subcarpetas:
            ruta_secuencia = os.path.join(ruta_paciente_dicom, sub)
            if os.path.isdir(ruta_secuencia):
                try:
                    # Convertimos la carpeta DICOM a un archivo .nii.gz
                    # El archivo tomará el nombre de la subcarpeta (FLAIR, SEG, etc.)
                    dicom2nifti.convert_directory(ruta_secuencia, ruta_paciente_nifti, compression=True, reorient=True)
                    print(f"  ✅ {sub} convertido con éxito.")
                except Exception as e:
                    print(f"  ❌ Error en {sub}: {e}")

if __name__ == "__main__":
    # Probamos con los primeros 5 para verificar
    convertir_training_set(num_pacientes=5)
    print("\n--- Proceso finalizado ---")