import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import glob

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from entrenamiento.dataset_loader import cargar_datos
from modelos.modelo_unet import crear_unet

DATASET_PATH    = os.path.join(BASE_DIR, "dataset")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "modelos", "modelo_tumores.pth")


# ── MEJORA 1 — Dice + BCE Loss ────────────────────────────────────────────────
# BCELoss plana ignora el desbalance extremo (tejido sano >> tumor).
# Dice Loss penaliza directamente las discrepancias en la forma de la máscara.
# Combinadas son el estándar en segmentación médica.

def dice_loss(pred, target, smooth=1.0):
    """Dice Loss diferenciable para segmentación binaria."""
    pred_flat   = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1.0 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def dice_bce_loss(pred, target, bce_weight=0.5):
    """
    Pérdida combinada Dice + BCE ponderada.
    bce_weight controla el balance: 0 → solo Dice, 1 → solo BCE.
    0.5 es el valor estándar recomendado en la literatura.
    """
    # MEJORA 2 — pos_weight para corregir desbalance de clases
    # Ratio aprox. 20:1 (fondo vs tumor) en imágenes cerebrales BraTS.
    pos_weight = torch.tensor([20.0], device=pred.device)
    bce = F.binary_cross_entropy_with_logits(
        torch.logit(pred.clamp(1e-6, 1 - 1e-6)),
        target,
        pos_weight=pos_weight
    )
    dl  = dice_loss(pred, target)
    return bce_weight * bce + (1.0 - bce_weight) * dl


# ── MEJORA 3 — Dice Score para monitorear calidad de segmentación ─────────────
# val_loss baja pero Dice Score bajo significa que el modelo predice todo como fondo.
# Esta métrica lo hace visible inmediatamente.

def dice_score(pred_bin, target, smooth=1.0):
    """Dice Score (F1 sobre máscaras binarias) en [0, 1]."""
    pred_flat   = pred_bin.view(-1).float()
    target_flat = target.view(-1).float()
    intersection = (pred_flat * target_flat).sum()
    return ((2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)).item()


# ── Utilidades ────────────────────────────────────────────────────────────────

def obtener_rutas_pacientes(path):
    return [d for d in glob.glob(os.path.join(path, "BraTS2021_*")) if os.path.isdir(d)]


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def ejecutar_entrenamiento():
    print("\n" + "=" * 55)
    print("  SISTEMA DE ENTRENAMIENTO U-NET — PYTORCH")
    print("=" * 55)

    # 1. Rutas de pacientes
    rutas_totales      = obtener_rutas_pacientes(DATASET_PATH)
    if not rutas_totales:
        print(f"[ERROR] No se encontraron pacientes en: {DATASET_PATH}")
        return

    rutas = rutas_totales
    print(f"Usando todos los pacientes disponibles: {len(rutas)}")

    # 2. Split train / val
    rutas_train, rutas_val = train_test_split(rutas, test_size=0.2, random_state=42)

    # MEJORA 4 — augmentar=True solo en entrenamiento
    print(f"\n[1/3] Cargando train ({len(rutas_train)} pacientes, con augmentación)...")
    X_train, y_train = cargar_datos(rutas_train, augmentar=True)

    print(f"[2/3] Cargando validación ({len(rutas_val)} pacientes, sin augmentación)...")
    X_val, y_val = cargar_datos(rutas_val, augmentar=False)

    print(f"\n[3/3] Dataset listo:")
    print(f"      Train : {X_train.shape[0]:>6} imágenes")
    print(f"      Val   : {X_val.shape[0]:>6} imágenes")

    # 3. Convertir a tensores PyTorch (B, C, H, W)
    def to_tensor(arr):
        return torch.tensor(arr).permute(0, 3, 1, 2).float()

    train_loader = DataLoader(
        TensorDataset(to_tensor(X_train), to_tensor(y_train)),
        batch_size=16, shuffle=True,  num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(to_tensor(X_val), to_tensor(y_val)),
        batch_size=16, shuffle=False, num_workers=0, pin_memory=True
    )

    # 4. Modelo y optimizador
    model, device = crear_unet()
    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=4, factor=0.5
    )

    # 5. Bucle de entrenamiento
    print("\n--- COMENZANDO ENTRENAMIENTO ---\n")
    mejor_val_loss    = float('inf')
    mejor_dice        = 0.0
    paciencia_contador = 0
    PACIENCIA = 16
    EPOCHS    = 100

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    for epoch in range(1, EPOCHS + 1):

        # ── TRAIN ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = dice_bce_loss(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── VALIDACIÓN ─────────────────────────────────────────────────────
        model.eval()
        val_loss  = 0.0
        val_dice  = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                pred     = model(X_batch)
                val_loss += dice_bce_loss(pred, y_batch).item()

                # MEJORA 3 — Dice Score en validación
                pred_bin  = (pred > 0.5).float()
                val_dice += dice_score(pred_bin, y_batch)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:>3}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f}")

        # ── Guardar mejor modelo (por val_loss) ────────────────────────────
        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            mejor_dice     = val_dice
            paciencia_contador = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"         [GUARDADO] val_loss={val_loss:.4f}  Dice={val_dice:.4f}")
        else:
            paciencia_contador += 1
            if paciencia_contador >= PACIENCIA:
                print(f"\n[STOP] Early stopping tras {PACIENCIA} epochs sin mejora.")
                break

    print(f"\n[OK] Entrenamiento finalizado.")
    print(f"     Mejor val_loss : {mejor_val_loss:.4f}")
    print(f"     Mejor Dice     : {mejor_dice:.4f}")
    print(f"     Modelo en      : {CHECKPOINT_PATH}")


if __name__ == "__main__":
    ejecutar_entrenamiento()