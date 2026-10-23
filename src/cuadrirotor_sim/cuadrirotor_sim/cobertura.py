"""Ruta de barrido y mapa de cobertura de la búsqueda.

  ruta_barrido     franjas paralelas (boustrophedon) que cubren un rectángulo.
  ruta_serpentina  serpentina que avanza hacia el frente del dron: barridos de
                   lado a lado unidos por semicírculos, muestreada cada 'paso' m.
  Cobertura      malla de celdas de la zona; marca las que ya vio la cámara
                 proyectando una rejilla de rayos de la imagen sobre el terreno.
"""
import math

import numpy as np


def ruta_barrido(zona, separacion, inicio):
    """Vértices de un recorrido en franjas a lo largo de x que cubre la zona.

    zona = (xmin, xmax, ymin, ymax). Las franjas van a separacion/2 de los bordes y
    el recorrido empieza en la esquina más cercana a 'inicio'.
    """
    xmin, xmax, ymin, ymax = zona
    n = max(1, math.ceil((ymax - ymin) / separacion))
    paso = (ymax - ymin) / n
    ys = [ymin + paso * (k + 0.5) for k in range(n)]
    x0, x1 = xmin + paso / 2, xmax - paso / 2
    if abs(inicio[1] - ys[0]) > abs(inicio[1] - ys[-1]):
        ys = ys[::-1]
    izquierda = abs(inicio[0] - x0) <= abs(inicio[0] - x1)
    ruta = []
    for y in ys:
        ruta += [(x0, y), (x1, y)] if izquierda else [(x1, y), (x0, y)]
        izquierda = not izquierda
    return np.array(ruta)


def ruta_serpentina(zona, separacion, inicio, rumbo, paso=5.0):
    """Serpentina que cubre la zona avanzando en el eje más parecido a 'rumbo'.

    Los barridos son perpendiculares al avance, separados ~'separacion' y a
    separacion/2 de los bordes; cada uno se une con el siguiente por un
    semicírculo de radio separacion/2 (sin esquinas: el dron no tiene que frenar).
    Empieza en el lado más cercano a 'inicio'. Regresa los puntos (N x 2) cada
    ~'paso' metros.
    """
    xmin, xmax, ymin, ymax = zona
    eje = 0 if abs(math.cos(rumbo)) >= abs(math.sin(rumbo)) else 1   # 0: avanza en x
    sentido = math.copysign(1.0, math.cos(rumbo) if eje == 0 else math.sin(rumbo))
    fmin, fmax = (xmin, xmax) if eje == 0 else (ymin, ymax)
    lmin, lmax = (ymin, ymax) if eje == 0 else (xmin, xmax)
    n = max(1, math.ceil((fmax - fmin) / separacion))
    p = (fmax - fmin) / n
    r = p / 2
    f0 = fmin + p / 2 if sentido > 0 else fmax - p / 2
    la, lb = lmin + p / 2, lmax - p / 2
    l_inicio = inicio[1] if eje == 0 else inicio[0]
    if abs(l_inicio - la) > abs(l_inicio - lb):
        la, lb = lb, la
    lc = (lmin + lmax) / 2
    puntos = []
    for k in range(n):
        f = f0 + sentido * k * p
        m = max(2, math.ceil(abs(lb - la) / paso) + 1)
        puntos += [(f, l) for l in np.linspace(la, lb, m)][:-1]
        if k == n - 1:
            puntos.append((f, lb))
            break
        sigma = math.copysign(1.0, lb - lc)        # la vuelta sale hacia afuera
        m = max(2, math.ceil(math.pi * r / paso) + 1)
        for phi in np.linspace(0.0, math.pi, m)[:-1]:
            puntos.append((f + sentido * r * (1 - math.cos(phi)), lb + sigma * r * math.sin(phi)))
        la, lb = lb, la
    P = np.array(puntos)
    return P if eje == 0 else P[:, ::-1]


def rayos_camara(R_cam, f, ancho, alto, nu=24, nv=18):
    """Direcciones (mundo) de una rejilla nu x nv de píxeles de la cámara.

    El marco de la cámara tiene x hacia adelante, y a la izquierda y z arriba.
    """
    u = np.linspace(0.5, ancho - 0.5, nu)
    v = np.linspace(0.5, alto - 0.5, nv)
    U, V = np.meshgrid(u, v)
    d = np.stack([np.ones(U.size), -(U.ravel() - ancho / 2) / f, -(V.ravel() - alto / 2) / f], 1)
    d = d @ R_cam.T
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def cortar_rayos(terreno, o, dirs, alcance, paso=2.0):
    """Puntos donde cada rayo toca el suelo (NaN si no lo toca dentro del alcance)."""
    t = np.arange(paso, alcance + paso / 2, paso)
    P = o[None, None, :] + t[None, :, None] * dirs[:, None, :]       # rayos x muestras x 3
    bajo = P[..., 2] <= terreno.altura(P[..., 0], P[..., 1])
    toca = bajo.any(axis=1)
    k = np.argmax(bajo, axis=1)
    puntos = np.full((len(dirs), 3), np.nan)
    puntos[toca] = P[toca, k[toca]]
    return puntos


class Cobertura:

    def __init__(self, zona, celda=10.0):
        self.xmin, self.xmax, self.ymin, self.ymax = zona
        self.celda = celda
        self.nx = math.ceil((self.xmax - self.xmin) / celda)
        self.ny = math.ceil((self.ymax - self.ymin) / celda)
        self.vista = np.zeros((self.ny, self.nx), bool)

    def marcar(self, puntos):
        p = puntos[~np.isnan(puntos[:, 0])]
        ix = np.floor((p[:, 0] - self.xmin) / self.celda).astype(int)
        iy = np.floor((p[:, 1] - self.ymin) / self.celda).astype(int)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        self.vista[iy[ok], ix[ok]] = True

    def fraccion(self):
        return float(self.vista.mean())
