import torch
import torch.nn as nn


# ── Bloque base ───────────────────────────────────────────────────────────────

class DobleConv(nn.Module):
    """
    Dos convoluciones 3×3 con BatchNorm y ReLU.

    MEJORA: se añade un tercer parámetro `dropout_p` para poder
    aumentar la regularización en el bottleneck sin afectar el encoder.
    BatchNorm va ANTES de ReLU (orden estándar moderno).
    """

    def __init__(self, entrada, salida, dropout_p=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(entrada, salida, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(salida),
            nn.ReLU(inplace=True),
            nn.Conv2d(salida, salida, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(salida),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout2d(dropout_p))
        self.bloque = nn.Sequential(*layers)

    def forward(self, x):
        return self.bloque(x)


# ── U-Net ─────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net para segmentación binaria de tumores cerebrales (1 canal entrada, 1 salida).

    Cambios respecto a la versión original:
      • Cuatro niveles de encoder/decoder en lugar de tres → más capacidad
        para detectar tumores pequeños (patrones de alta frecuencia).
      • Dropout diferenciado: 0.1 en encoder, 0.3 en bottleneck.
      • bias=False en Conv2d (redundante con BatchNorm, reduce parámetros).
      • La salida NO aplica sigmoid aquí; se aplica en forward para que
        entrenar.py pueda usar BCEWithLogitsLoss si se necesita.
        Para inferencia se llama con torch.sigmoid(modelo(x)).
    """

    def __init__(self):
        super().__init__()
        self.pool    = nn.MaxPool2d(2)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.c1 = DobleConv(1,   32,  dropout_p=0.1)   # 128 → 64
        self.c2 = DobleConv(32,  64,  dropout_p=0.1)   # 64  → 32
        self.c3 = DobleConv(64,  128, dropout_p=0.1)   # 32  → 16
        self.c4 = DobleConv(128, 256, dropout_p=0.1)   # 16  → 8   (nuevo)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bn = DobleConv(256, 512, dropout_p=0.3)   # 8   → 8

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d1  = DobleConv(512, 256, dropout_p=0.1)  # skip c4

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d2  = DobleConv(256, 128, dropout_p=0.1)  # skip c3

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d3  = DobleConv(128, 64,  dropout_p=0.1)  # skip c2

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.d4  = DobleConv(64, 32,   dropout_p=0.0)  # skip c1

        # ── Salida ────────────────────────────────────────────────────────────
        # Sin sigmoid: se aplica externamente para compatibilidad con
        # BCEWithLogitsLoss durante entrenamiento e inferencia manual.
        self.salida = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.c1(x)
        s2 = self.c2(self.pool(s1))
        s3 = self.c3(self.pool(s2))
        s4 = self.c4(self.pool(s3))

        # Bottleneck
        bn = self.bn(self.pool(s4))

        # Decoder con skip connections
        x = self.d1(torch.cat([self.up1(bn), s4], dim=1))
        x = self.d2(torch.cat([self.up2(x),  s3], dim=1))
        x = self.d3(torch.cat([self.up3(x),  s2], dim=1))
        x = self.d4(torch.cat([self.up4(x),  s1], dim=1))

        # Sigmoid para obtener probabilidades en [0, 1]
        return torch.sigmoid(self.salida(x))


# ── Factory ───────────────────────────────────────────────────────────────────

def crear_unet():
    """Instancia la U-Net en el dispositivo disponible y la devuelve lista para usar."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[GPU] Modelo UNet cargado en: {device}  |  Parámetros: {n_params:,}")
    return model, device


# ── Test rápido ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, device = crear_unet()
    x   = torch.randn(2, 1, 128, 128).to(device)
    out = model(x)
    print(f"Entrada: {x.shape}  →  Salida: {out.shape}")
    assert out.shape == (2, 1, 128, 128), "Shape de salida incorrecto"
    assert out.min() >= 0 and out.max() <= 1, "Salida fuera de [0,1]"
    print("[OK] Test de forma y rango superado.")