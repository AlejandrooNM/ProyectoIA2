import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import glob

# Rutas
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from entrenamiento.dataset_loader import cargar_datos
from modelos.modelo_unet import crear_unet

DATASET_PATH = os.path.join(BASE_DIR, "dataset")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "modelos", "modelo_tumores.pth")

def obtener_rutas_pacientes(path):
    pacientes = [d for d in glob.glob(os.path.join(path, "BraTS2021_*")) if os.path.isdir(d)]
    return pacientes

def ejecutar_entrenamiento():
    print("\n" + "="*50)
    print("SISTEMA DE ENTRENAMIENTO U-NET — PYTORCH GPU")
    print("="*50)

    # 1. Rutas de pacientes
    rutas_totales = obtener_rutas_pacientes(DATASET_PATH)
    num_pacientes_limite = 500

    if not rutas_totales:
        print(f"Error: No se encontraron pacientes en {DATASET_PATH}")
        return

    if len(rutas_totales) > num_pacientes_limite:
        rutas = random.sample(rutas_totales, num_pacientes_limite)
        print(f"Total detectados: {len(rutas_totales)} | Usando: {num_pacientes_limite} al azar")
    else:
        rutas = rutas_totales
        print(f"Usando todos los pacientes disponibles: {len(rutas)}")

    # 2. Carga de datos
    rutas_train, rutas_val = train_test_split(rutas, test_size=0.2, random_state=42)

    print(f"\n[1/3] Cargando entrenamiento ({len(rutas_train)} pacientes)...")
    X_train, y_train = cargar_datos(rutas_train)

    print(f"[2/3] Cargando validación ({len(rutas_val)} pacientes)...")
    X_val, y_val = cargar_datos(rutas_val)

    print(f"\n[3/3] Dataset listo:")
    print(f" - Train: {X_train.shape[0]} imágenes")
    print(f" - Val:   {X_val.shape[0]} imágenes")

    # 3. Convertir a tensores PyTorch (B, C, H, W)
    X_train_t = torch.tensor(X_train).permute(0, 3, 1, 2).float()
    y_train_t = torch.tensor(y_train).permute(0, 3, 1, 2).float()
    X_val_t   = torch.tensor(X_val).permute(0, 3, 1, 2).float()
    y_val_t   = torch.tensor(y_val).permute(0, 3, 1, 2).float()

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=16, shuffle=False)

    # 4. Modelo
    model, device = crear_unet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5,)
    criterio  = nn.BCELoss()

    # 5. Entrenamiento
    print("\n--- COMENZANDO ENTRENAMIENTO ---")
    mejor_val_loss = float('inf')
    paciencia_contador = 0
    PACIENCIA = 16
    EPOCHS = 100

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterio(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- VALIDACIÓN ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                pred     = model(X_batch)
                val_loss += criterio(pred, y_batch).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # --- GUARDAR MEJOR MODELO ---
        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            paciencia_contador = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  [OK] Mejor modelo guardado (val_loss: {val_loss:.4f})")
        else:
            paciencia_contador += 1
            if paciencia_contador >= PACIENCIA:
                print(f"\n[STOP] Early stopping — sin mejora en {PACIENCIA} epochs.")
                break

    print(f"\n[OK] Entrenamiento finalizado.")
    print(f"Mejor modelo guardado en: {CHECKPOINT_PATH}")

if __name__ == "__main__":
    ejecutar_entrenamiento()