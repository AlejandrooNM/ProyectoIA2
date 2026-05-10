import torch
import torch.nn as nn

class DobleConv(nn.Module):
    """Dos convoluciones seguidas — bloque base de la U-Net"""
    def __init__(self, entrada, salida):
        super().__init__()
        self.bloque = nn.Sequential(
            nn.Conv2d(entrada, salida, kernel_size=3, padding=1),
            nn.BatchNorm2d(salida),
            nn.ReLU(inplace=True),
            nn.Conv2d(salida, salida, kernel_size=3, padding=1),
            nn.BatchNorm2d(salida),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.bloque(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ENCODER
        self.c1 = DobleConv(1, 32)
        self.c2 = DobleConv(32, 64)
        self.c3 = DobleConv(64, 128)

        # BOTTLENECK
        self.bn = DobleConv(128, 256)

        # DECODER
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c4  = DobleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c5  = DobleConv(128, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.c6  = DobleConv(64, 32)

        # SALIDA
        self.salida = nn.Conv2d(32, 1, kernel_size=1)

        self.pool    = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        # Encoder
        c1 = self.c1(x)
        c2 = self.c2(self.dropout(self.pool(c1)))
        c3 = self.c3(self.dropout(self.pool(c2)))

        # Bottleneck
        bn = self.bn(self.pool(c3))

        # Decoder
        u1 = self.up1(bn)
        u1 = self.c4(torch.cat([u1, c3], dim=1))

        u2 = self.up2(u1)
        u2 = self.c5(torch.cat([u2, c2], dim=1))

        u3 = self.up3(u2)
        u3 = self.c6(torch.cat([u3, c1], dim=1))

        return torch.sigmoid(self.salida(u3))


def crear_unet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet().to(device)
    print(f"[GPU] Modelo cargado en: {device}")
    return model, device


if __name__ == "__main__":
    model, device = crear_unet()
    x = torch.randn(1, 1, 128, 128).to(device)
    out = model(x)
    print(f"Salida: {out.shape}")