"""Evasión reactiva de obstáculos con el LiDAR y el relieve.

El relieve se conoce (DEM): el planificador vuela a 'altura_busqueda' sobre él
y no avanza mientras esté por debajo de 'altura_seguridad' sobre el terreno que
tiene adelante (primero sube). Lo que el DEM no tiene —árboles, rocas,
animales— lo ve el LiDAR:

  memoria    los puntos del LiDAR, en el marco del mundo, sin los que quedan a
             menos de 'suelo' del DEM (son el terreno), reducidos a vóxeles de
             1 m y olvidados más allá de 'memoria' metros del dron.
  corregir   si el tramo recto hacia el objetivo pasa a menos de 'radio' de
             algún punto a la altura del dron (±'banda'), arma un histograma
             polar (como VFH): un sector está bloqueado si un punto a menos de
             'alcance' cae dentro de él, ensanchado por asin(radio/d). Elige el
             sector libre más cercano al rumbo del objetivo (con algo de
             preferencia por el que eligió antes, para no oscilar) y devuelve un
             punto a 'paso' metros en esa dirección. Sin sectores libres, sube.
             Al bajar, no baja más que 'radio' sobre el punto más alto que
             tenga debajo. Para no oscilar entre desviarse y no, mientras está
             desviado solo vuelve al tramo recto si queda libre con 'histeresis'
             veces el radio.
"""
import math

import numpy as np


def envolver(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Evasion:

    def __init__(self, terreno, radio=3.0, alcance=20.0, banda=3.0, paso=8.0,
                 suelo=1.5, memoria=60.0, voxel=1.0, sectores=72, subida=4.0, histeresis=1.4):
        self.terreno = terreno
        self.radio, self.alcance, self.banda, self.paso = radio, alcance, banda, paso
        self.suelo, self.memoria, self.voxel = suelo, memoria, voxel
        self.subida, self.histeresis = subida, histeresis
        self.th = np.linspace(-np.pi, np.pi, sectores, endpoint=False)
        self.medio_sector = np.pi / sectores
        self.puntos = np.zeros((0, 3))
        self.rumbo_previo = None
        self.activa = False      # True mientras se desvía del tramo recto

    # --------------------------------------------------------------- memoria
    def agregar(self, puntos, centro):
        """Agrega puntos (mundo, N x 3) y olvida los lejanos de 'centro'."""
        p = puntos[np.all(np.isfinite(puntos), axis=1)]
        p = p[p[:, 2] > self.terreno.altura(p[:, 0], p[:, 1]) + self.suelo]
        P = np.vstack([self.puntos, p])
        P = P[np.hypot(P[:, 0] - centro[0], P[:, 1] - centro[1]) < self.memoria]
        if len(P):
            _, i = np.unique(np.floor(P / self.voxel).astype(np.int64), axis=0, return_index=True)
            P = P[i]
        self.puntos = P

    def techo(self, xy, radio=None):
        """Altura del punto más alto a menos de 'radio' (horizontal) de xy, o -inf."""
        r = self.radio if radio is None else radio
        P = self.puntos
        cerca = np.hypot(P[:, 0] - xy[0], P[:, 1] - xy[1]) < r
        return float(P[cerca, 2].max()) if cerca.any() else -math.inf

    # -------------------------------------------------------------- evasión
    def _bloqueado(self, Q, u, largo, radio):
        """¿Algún punto (xy relativo) a menos de 'radio' del tramo 0 -> largo*u?"""
        s = np.clip(Q @ u, 0.0, largo)
        return bool(np.any(np.linalg.norm(Q - s[:, None] * u, axis=1) < radio))

    def corregir(self, c, g):
        """Objetivo seguro desde la consigna c hacia el objetivo g (ambos 3D)."""
        g = np.array(g, float)
        # Al bajar: no meterse en las copas.
        if g[2] < c[2]:
            g[2] = max(g[2], self.techo(c[:2]) + self.radio)
        rel = self.puntos - c
        cerca = (np.abs(rel[:, 2]) < self.banda) & (np.hypot(rel[:, 0], rel[:, 1]) < self.alcance)
        Q = rel[cerca, :2]
        d = g[:2] - c[:2]
        dist = float(np.linalg.norm(d))
        if len(Q) == 0 or dist < 0.5:
            self.activa, self.rumbo_previo = False, None
            return g
        u = d / dist
        radio = self.radio * (self.histeresis if self.activa else 1.0)
        if not self._bloqueado(Q, u, min(dist, self.alcance), radio):
            self.activa, self.rumbo_previo = False, None
            return g
        # Histograma polar: sectores bloqueados por cada punto ensanchado por asin(radio/d).
        r = np.linalg.norm(Q, axis=1)
        ang = np.arctan2(Q[:, 1], Q[:, 0])
        ancho = np.arcsin(np.minimum(1.0, self.radio / np.maximum(r, 1e-3))) + self.medio_sector
        bloq = (np.abs(envolver(self.th[:, None] - ang[None, :])) < ancho[None, :]).any(axis=1)
        self.activa = True
        if bloq.all():
            self.rumbo_previo = None
            return np.array([c[0], c[1], c[2] + self.subida])
        rumbo_g = math.atan2(u[1], u[0])
        costo = np.abs(envolver(self.th - rumbo_g))
        if self.rumbo_previo is not None:
            costo = costo + 0.3 * np.abs(envolver(self.th - self.rumbo_previo))
        costo[bloq] = np.inf
        rumbo = float(self.th[np.argmin(costo)])
        self.rumbo_previo = rumbo
        paso = min(self.paso, dist)
        return np.array([c[0] + paso * math.cos(rumbo), c[1] + paso * math.sin(rumbo), g[2]])
