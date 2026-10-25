"""Bloques de construcción de FuegoNet, la red tipo YOLO26 para fuego y humo.

Cada bloque es una pieza de la familia YOLO escrita desde cero con PyTorch puro,
sin depender de ultralytics:

  Conv        convolución + BatchNorm + SiLU, la unidad mínima.
  DWConv      convolución en profundidad (un filtro por canal), barata.
  Cuello      «bottleneck» residual: dos Conv 3x3 y un atajo x + f(x).
  C3k         CSP con cuellos de botella de kernel k (dos ramas, una se procesa).
  C3k2        el bloque principal de YOLO11/YOLO26: divide los canales, apila
              n subbloques y concatena todas las salidas intermedias.
  SPPF        agrupamiento piramidal rápido: tres max-pool 5x5 en cadena.
  Atencion    autoatención multicabeza sobre el mapa de características.
  BloquePSA   atención + red de avance con atajos (como un transformador).
  C2PSA       CSP con bloques PSA, al final de la columna vertebral.
"""
import math

import torch
from torch import nn


def relleno(k, d=1):
    """Relleno que conserva el tamaño espacial con kernel k y dilatación d."""
    k = d * (k - 1) + 1 if d > 1 else k
    return k // 2


class Conv(nn.Module):
    """Conv2d sin sesgo + BatchNorm2d + SiLU (o identidad si act=False)."""

    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, relleno(k, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    """Convolución en profundidad: grupos = máximo común divisor de los canales."""

    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Cuello(nn.Module):
    """Bottleneck: y = x + Conv(Conv(x)) si hay atajo y los canales coinciden."""

    def __init__(self, c1, c2, atajo=True, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0])
        self.cv2 = Conv(c_, c2, k[1])
        self.suma = atajo and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.suma else y


class C3k(nn.Module):
    """CSP de tres convoluciones: una rama pasa por n cuellos, la otra no."""

    def __init__(self, c1, c2, n=2, atajo=True, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c1, c_, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Cuello(c_, c_, atajo, (k, k), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k2(nn.Module):
    """Bloque C3k2 (C2f con subbloques intercambiables).

    La entrada se proyecta a 2c canales y se parte en dos mitades. La segunda
    mitad pasa por n subbloques en cadena y se guarda cada salida; al final se
    concatenan las dos mitades y las n salidas ((2 + n) c canales) y una Conv 1x1
    las mezcla. Así el gradiente tiene caminos cortos hacia todas las capas.

    El subbloque es un Cuello, un C3k (c3k=True) o un Cuello seguido de
    atención (atencion=True), como en la última etapa de YOLO26.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, atencion=False, atajo=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        if atencion:
            sub = lambda: nn.Sequential(Cuello(self.c, self.c, atajo),
                                        BloquePSA(self.c, cabezas=max(self.c // 64, 1)))
        elif c3k:
            sub = lambda: C3k(self.c, self.c, 2, atajo)
        else:
            sub = lambda: Cuello(self.c, self.c, atajo)
        self.m = nn.ModuleList(sub() for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast.

    Tres max-pool 5x5 en cadena equivalen a ventanas de 5, 9 y 13 píxeles: la
    red ve el contexto a varias escalas sin costo de parámetros. Con atajo=True
    (YOLO26) se suma la entrada a la salida.
    """

    def __init__(self, c1, c2, k=5, n=3, atajo=False):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1)
        self.m = nn.MaxPool2d(k, 1, k // 2)
        self.n = n
        self.suma = atajo and c1 == c2

    def forward(self, x):
        y = [self.cv1(x)]
        for _ in range(self.n):
            y.append(self.m(y[-1]))
        y = self.cv2(torch.cat(y, 1))
        return x + y if self.suma else y


class Atencion(nn.Module):
    """Autoatención multicabeza con convoluciones 1x1 y codificación posicional.

    Cada píxel del mapa (N = H*W) es un «token». Se calculan consultas q,
    llaves k y valores v; el peso entre dos píxeles es softmax(q·k/sqrt(dk)).
    Una convolución 3x3 en profundidad sobre v hace de codificación posicional.
    """

    def __init__(self, dim, cabezas=8, razon=0.5):
        super().__init__()
        self.cabezas = cabezas
        self.dc = dim // cabezas            # dimensión de cada cabeza (valores)
        self.dk = int(self.dc * razon)      # dimensión de consultas y llaves
        self.escala = self.dk ** -0.5
        self.qkv = Conv(dim, dim + 2 * self.dk * cabezas, 1, act=False)
        self.proy = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).view(B, self.cabezas, 2 * self.dk + self.dc, H * W).split(
            [self.dk, self.dk, self.dc], dim=2)
        a = ((q * self.escala).transpose(-2, -1) @ k).softmax(dim=-1)   # (B, h, N, N)
        y = (v @ a.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proy(y)


class BloquePSA(nn.Module):
    """Atención y red de avance (FFN), cada una con su conexión residual."""

    def __init__(self, c, razon=0.5, cabezas=4, atajo=True):
        super().__init__()
        self.atn = Atencion(c, cabezas, razon)
        self.ffn = nn.Sequential(Conv(c, 2 * c, 1), Conv(2 * c, c, 1, act=False))
        self.suma = atajo

    def forward(self, x):
        x = x + self.atn(x) if self.suma else self.atn(x)
        return x + self.ffn(x) if self.suma else self.ffn(x)


class C2PSA(nn.Module):
    """CSP con n bloques PSA: la atención se aplica solo a la mitad de los canales."""

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(BloquePSA(self.c, 0.5, max(self.c // 64, 1)) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((a, self.m(b)), 1))
