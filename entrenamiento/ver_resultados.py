import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataset_loader import cargar_datos
import os

# 1. Cargar el modelo que acabas de entrenar
MODEL_PATH = r"E:\ProyectoIA2\modelos\modelo_tumores.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# 2. Cargar un par de pacientes nuevos para la prueba
# Usamos solo 2 pacientes para que sea rápido
DATASET_PATH = r"E:\ProyectoIA2\dataset"
X_test, y_test = cargar_datos(DATASET_PATH, num_pacientes=2)

# 3. Hacer la predicción
# La IA intentará adivinar dónde está el tumor en X_test
predicciones = model.predict(X_test)

# 4. Graficar los resultados
def visualizar_prediccion(indice):
    plt.figure(figsize=(12, 4))

    # Imagen original (T1wCE)
    plt.subplot(1, 3, 1)
    plt.title("Original (T1wCE)")
    plt.imshow(X_test[indice].reshape(128, 128), cmap='gray')
    plt.axis('off')

    # Máscara Real (Lo que descargaste de Kaggle)
    plt.subplot(1, 3, 2)
    plt.title("Tumor Real (Manual)")
    plt.imshow(y_test[indice].reshape(128, 128), cmap='Reds', alpha=0.5)
    plt.axis('off')

    # Predicción de la IA (Lo que tu modelo cree que es tumor)
    plt.subplot(1, 3, 3)
    plt.title("Predicción de la IA")
    # Aplicamos un umbral de 0.5 para binarizar la predicción
    umbral_pred = (predicciones[indice] > 0.5).astype(np.float32)
    plt.imshow(umbral_pred.reshape(128, 128), cmap='Blues', alpha=0.5)
    plt.axis('off')

    plt.show()

# Visualizamos una rebanada al azar de las que cargamos
visualizar_prediccion(np.random.randint(0, len(X_test)))