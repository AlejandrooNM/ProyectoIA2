import os
# Configuramos el backend antes de importar keras
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers, models

def crear_unet(input_size=(128, 128, 1)):
    # Definición de la entrada
    inputs = layers.Input(shape=input_size)

    # --- ENCODER (Contracción) ---
    # Bloque 1
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # Bloque 2
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # --- BOTTLENECK (El punto más profundo) ---
    b1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    b1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(b1)

    # --- DECODER (Expansión) ---
    # Bloque 3 (Sube y concatena con el bloque 2)
    u1 = layers.UpSampling2D((2, 2))(b1)
    u1 = layers.concatenate([u1, c2])
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c3)

    # Bloque 4 (Sube y concatena con el bloque 1)
    u2 = layers.UpSampling2D((2, 2))(c3)
    u2 = layers.concatenate([u2, c1])
    c4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u2)
    c4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c4)

    # --- SALIDA ---
    # Capa final con activación Sigmoid para segmentación binaria (Tumor vs Fondo)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c4)

    # Crear el modelo
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compilación
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# Prueba rápida para verificar que el modelo se construye correctamente
if __name__ == "__main__":
    unet = crear_unet()
    unet.summary()