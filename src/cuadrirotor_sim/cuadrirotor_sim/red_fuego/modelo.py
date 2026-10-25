"""FuegoNet: detector de una etapa con la arquitectura de YOLO26.

Tres partes, igual que YOLO26:
  columna vertebral  (backbone)  extrae características a 1/8, 1/16 y 1/32;
  cuello             (neck, PAN-FPN) mezcla las tres escalas de arriba abajo y
                     de abajo arriba;
  cabeza             (head) predice, en cada celda de cada escala, 4 distancias
                     a los lados de la caja (l, t, r, b) y una puntuación por clase.

Lo propio de YOLO26 que se conserva:
  * reg_max = 1: las distancias se regresan directo, sin la distribución DFL
    de YOLOv8/YOLO11 (más simple y más rápido de exportar).
  * Cabeza doble: «uno a muchos» (o2m) solo para entrenar y «uno a uno» (o2o)
    para inferir. La o2o aprende a dar UNA caja por objeto, así que no hace
    falta NMS: basta tomar las k puntuaciones más altas.

Las escalas siguen el yolo26.yaml de ultralytics ([profundidad, anchura, máx]).
"""
import copy
import math

import torch
from torch import nn

from .bloques import C2PSA, C3k2, SPPF, Conv, DWConv

ESCALAS = {'n': (0.50, 0.25, 1024), 's': (0.50, 0.50, 1024), 'm': (0.50, 1.00, 512)}
PASOS = (8, 16, 32)


def anclas(mapas, pasos, desfase=0.5):
    """Centros de las celdas (en unidades de celda) y el paso de cada una."""
    puntos, paso = [], []
    for m, s in zip(mapas, pasos):
        _, _, h, w = m.shape
        sy, sx = torch.meshgrid(torch.arange(h, device=m.device, dtype=m.dtype) + desfase,
                                torch.arange(w, device=m.device, dtype=m.dtype) + desfase,
                                indexing='ij')
        puntos.append(torch.stack((sx, sy), -1).view(-1, 2))
        paso.append(torch.full((h * w, 1), s, device=m.device, dtype=m.dtype))
    return torch.cat(puntos), torch.cat(paso)


def dist_a_caja(dist, puntos):
    """(l, t, r, b) respecto al centro de la celda -> (x1, y1, x2, y2)."""
    lt, rb = dist.chunk(2, -1)
    return torch.cat((puntos - lt, puntos + rb), -1)


class Cabeza(nn.Module):
    """Cabeza de detección sin NMS (Detect de YOLO26 con end2end=True)."""

    def __init__(self, nc, canales):
        super().__init__()
        self.nc, self.nl = nc, len(canales)
        self.pasos = PASOS
        c2 = max(16, canales[0] // 4, 4)
        c3 = max(canales[0], min(nc, 100))
        # Rama de cajas: dos Conv 3x3 y una 1x1 que da 4 números (reg_max = 1).
        self.caja = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4, 1)) for x in canales)
        # Rama de clases: convoluciones en profundidad + 1x1 (más ligera).
        self.clase = nn.ModuleList(
            nn.Sequential(nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                          nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                          nn.Conv2d(c3, nc, 1)) for x in canales)
        # Copias para la asignación uno a uno.
        self.caja_o2o = copy.deepcopy(self.caja)
        self.clase_o2o = copy.deepcopy(self.clase)
        self.max_det = 300

    def iniciar_sesgos(self, imgsz=640):
        """Sesgo de clase = log(prior) para que al inicio casi todo sea fondo."""
        for ramas in ((self.caja, self.clase), (self.caja_o2o, self.clase_o2o)):
            for a, b, s in zip(*ramas, self.pasos):
                a[-1].bias.data[:] = 2.0
                b[-1].bias.data[:] = math.log(5 / self.nc / (imgsz / s) ** 2)

    def _ramas(self, x, caja, clase):
        bs = x[0].shape[0]
        d = torch.cat([caja[i](x[i]).view(bs, 4, -1) for i in range(self.nl)], -1)
        c = torch.cat([clase[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], -1)
        return {'dist': d.permute(0, 2, 1), 'logits': c.permute(0, 2, 1), 'mapas': x}

    def forward(self, x):
        if not self.training:   # al inferir solo se usa la o2o: la o2m ni se calcula
            return self.decodificar(self._ramas(x, self.caja_o2o, self.clase_o2o))
        o2m = self._ramas(x, self.caja, self.clase)
        # La rama o2o ve las características desconectadas del grafo: su gradiente
        # no modifica la columna vertebral (la entrena solo la rama o2m).
        o2o = self._ramas([xi.detach() for xi in x], self.caja_o2o, self.clase_o2o)
        return {'o2m': o2m, 'o2o': o2o}

    def decodificar(self, p):
        """Salida (B, max_det, 6): x1, y1, x2, y2 en píxeles, confianza, clase."""
        puntos, paso = anclas(p['mapas'], self.pasos)
        cajas = dist_a_caja(p['dist'], puntos) * paso
        prob = p['logits'].sigmoid()
        conf, clase = prob.max(-1)
        k = min(self.max_det, conf.shape[1])
        conf, idx = conf.topk(k, dim=1)
        cajas = cajas.gather(1, idx[..., None].expand(-1, -1, 4))
        clase = clase.gather(1, idx)
        return torch.cat((cajas, conf[..., None], clase[..., None].to(cajas.dtype)), -1)


class FuegoNet(nn.Module):
    """Columna vertebral + cuello PAN + cabeza sin NMS, como yolo26.yaml."""

    def __init__(self, nc=2, escala='n'):
        super().__init__()
        prof, anch, cmax = ESCALAS[escala]
        c = lambda x: int(math.ceil(min(x, cmax) * anch / 8) * 8)   # canales múltiplos de 8
        n = lambda x: max(round(x * prof), 1)                        # repeticiones
        c3k = escala in 'ml'                                         # como ultralytics
        self.nc, self.escala = nc, escala
        # --- Columna vertebral ------------------------------------------------
        self.b0 = Conv(3, c(64), 3, 2)                                   # P1/2
        self.b1 = Conv(c(64), c(128), 3, 2)                              # P2/4
        self.b2 = C3k2(c(128), c(256), n(2), c3k, 0.25)
        self.b3 = Conv(c(256), c(256), 3, 2)                             # P3/8
        self.b4 = C3k2(c(256), c(512), n(2), c3k, 0.25)                  # -> P3
        self.b5 = Conv(c(512), c(512), 3, 2)                             # P4/16
        self.b6 = C3k2(c(512), c(512), n(2), True)                       # -> P4
        self.b7 = Conv(c(512), c(1024), 3, 2)                            # P5/32
        self.b8 = C3k2(c(1024), c(1024), n(2), True)
        self.b9 = SPPF(c(1024), c(1024), 5, 3, atajo=True)
        self.b10 = C2PSA(c(1024), c(1024), n(2))                         # -> P5
        # --- Cuello: de arriba abajo (FPN) -------------------------------------
        self.sube = nn.Upsample(None, 2, 'nearest')
        self.h13 = C3k2(c(1024) + c(512), c(512), n(2), True)
        self.h16 = C3k2(c(512) + c(512), c(256), n(2), True)             # salida P3
        # --- Cuello: de abajo arriba (PAN) -------------------------------------
        self.h17 = Conv(c(256), c(256), 3, 2)
        self.h19 = C3k2(c(256) + c(512), c(512), n(2), True)             # salida P4
        self.h20 = Conv(c(512), c(512), 3, 2)
        self.h22 = C3k2(c(512) + c(1024), c(1024), 1, True, 0.5, atencion=True)  # P5
        # --- Cabeza ------------------------------------------------------------
        self.cabeza = Cabeza(nc, (c(256), c(512), c(1024)))
        self.cabeza.iniciar_sesgos()

    def forward(self, x):
        x = self.b2(self.b1(self.b0(x)))
        p3 = self.b4(self.b3(x))
        p4 = self.b6(self.b5(p3))
        p5 = self.b10(self.b9(self.b8(self.b7(p4))))
        t4 = self.h13(torch.cat((self.sube(p5), p4), 1))
        s3 = self.h16(torch.cat((self.sube(t4), p3), 1))
        s4 = self.h19(torch.cat((self.h17(s3), t4), 1))
        s5 = self.h22(torch.cat((self.h20(s4), p5), 1))
        return self.cabeza([s3, s4, s5])


def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters())


if __name__ == '__main__':
    for e in ESCALAS:
        m = FuegoNet(2, e).eval()
        with torch.no_grad():
            y = m(torch.zeros(1, 3, 640, 640))
        print(f'FuegoNet-{e}: {contar_parametros(m) / 1e6:.2f} M parámetros, salida {tuple(y.shape)}')
