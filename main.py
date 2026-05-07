import sys
import os
import matplotlib.pyplot as plt

# 1. Configuración de rutas para que Python encuentre los módulos
# Obtenemos la ruta absoluta de la carpeta donde está este main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Intentamos importar los módulos locales de las subcarpetas
try:
    # Usamos imports directos ya que BASE_DIR está en el path
    from analisis.predictor import predecir_tumor
    from analisis.analizador import analizar_tumor
    from reportes.reporte import generar_reporte
except ImportError as e:
    print(f" Error de importación: {e}")
    print("Asegúrate de que las carpetas 'analisis' y 'reportes' tengan un archivo __init__.py")
    sys.exit(1)

def iniciar_programa():
    print("--- SISTEMA DE DETECCIÓN DE TUMORES CEREBRALES ---")
    
    # 2. Entrada de usuario
    ruta = input("Ingrese la ruta del archivo .nii.gz (o arrastre el archivo aquí): ").strip('"')

    if not os.path.exists(ruta):
        print(f"Error: No se encuentra el archivo en: {ruta}")
        return

    print("\nProcesando resonancia magnética con U-Net...")
    
    # 3. Flujo principal
    try:
        # original: corte MRI 2D
        # mascara: predicción binaria 2D
        original, mascara = predecir_tumor(ruta)

        # 4. Análisis estadístico
        resultado = analizar_tumor(mascara)

        print("\n===== DIAGNÓSTICO PRELIMINAR =====")
        for clave, valor in resultado.items():
            print(f"{clave.replace('_', ' ').capitalize()}: {valor}")

        # 5. Generación de PDF profesional
        generar_reporte(resultado)

        # 6. Visualización gráfica
        print("\nGenerando visualización médica...")
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title("MRI Original (Corte Central)")
        plt.imshow(original, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Segmentación IA (Tumor)")
        # Mostramos la máscara original en gris y encima el tumor en color 'jet'
        plt.imshow(original, cmap='gray')
        plt.imshow(mascara.squeeze(), cmap='jet', alpha=0.5) 
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Ocurrió un error durante el procesamiento: {e}")

if __name__ == "__main__":
    iniciar_programa()