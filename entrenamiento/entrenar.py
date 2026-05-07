import sys
import os

# 1. Configuración de rutas (Path)
# Esto permite que Python vea 'modelos' y también la carpeta actual 'entrenamiento'
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from sklearn.model_selection import train_test_split
from modelos.modelo_unet import crear_unet
# IMPORTANTE: Quitamos el punto (.) para que cargue directamente desde la carpeta
from dataset_loader import cargar_datos 

# Ruta relativa a la raíz del proyecto
DATASET_PATH = os.path.join(BASE_DIR, "dataset")

def ejecutar_entrenamiento():
    print("Iniciando proceso de entrenamiento...")
    
    # 2. Carga de datos
    X, y = cargar_datos(DATASET_PATH)

    if len(X) == 0:
        print(f"Error: No se cargaron imágenes. Verifica que la carpeta '{DATASET_PATH}' contenga pacientes con archivos Flair y Seg.")
        return

    print(f"Dataset cargado: {len(X)} imágenes preparadas.")

    # 3. División del dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Construcción y entrenamiento
    # Asegúrate de haber instalado tensorflow: pip install tensorflow
    model = crear_unet()

    print("--- Comenzando el entrenamiento ---")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=4
    )

    # 5. Guardado del modelo
    output_dir = os.path.join(BASE_DIR, "modelos")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_path = os.path.join(output_dir, "modelo_tumores.keras")
    model.save(save_path)
    print(f"Modelo guardado exitosamente en: {save_path}")

if __name__ == "__main__":
    ejecutar_entrenamiento()