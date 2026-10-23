"""Modelo digital de elevación (DEM) del terreno de un mundo.

El archivo .npz (lo genera scripts/generar_bosque_grande.py) guarda una malla
regular de alturas h[iy, ix] en los puntos (x0 + ix*paso, y0 + iy*paso). La
malla de Gazebo usa los mismos vértices y parte cada celda en dos triángulos por
la diagonal (ix, iy)-(ix+1, iy+1); aquí se interpola igual, así que la altura
coincide con la superficie que se ve y con la que choca el dron.

Sin archivo, el terreno es el plano z = 0 (mundos 'incendio' y 'bosque').
"""
import math

import numpy as np


class Terreno:

    def __init__(self, ruta=''):
        self.plano = not ruta
        if self.plano:
            return
        d = np.load(ruta)
        self.x0, self.y0, self.paso = float(d['x0']), float(d['y0']), float(d['paso'])
        self.h = d['h'].astype(float)
        self.ny, self.nx = self.h.shape

    def altura(self, x, y):
        """Altura del suelo en (x, y); acepta escalares o arreglos."""
        if self.plano:
            return np.zeros_like(np.asarray(x, float)) if np.ndim(x) else 0.0
        fx = np.clip((np.asarray(x, float) - self.x0) / self.paso, 0, self.nx - 1 - 1e-9)
        fy = np.clip((np.asarray(y, float) - self.y0) / self.paso, 0, self.ny - 1 - 1e-9)
        ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
        u, v = fx - ix, fy - iy
        h00, h10 = self.h[iy, ix], self.h[iy, ix + 1]
        h01, h11 = self.h[iy + 1, ix], self.h[iy + 1, ix + 1]
        z = np.where(u >= v, h00 + u * (h10 - h00) + v * (h11 - h10),
                     h00 + v * (h01 - h00) + u * (h11 - h01))
        return float(z) if np.ndim(z) == 0 else z

    def maximo_en_tramo(self, a, b, margen=0.0, n=25):
        """Altura máxima del suelo a lo largo del segmento a-b (xy), ensanchado por margen."""
        t = np.linspace(0.0, 1.0, n)
        x, y = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        if margen > 0:
            ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
            x = (x[:, None] + margen * np.cos(ang)).ravel()
            y = (y[:, None] + margen * np.sin(ang)).ravel()
        return float(np.max(self.altura(x, y)))

    def cortar_rayo(self, o, d, alcance=400.0, desfase=0.0):
        """Primer punto del rayo o + t d (t > 0) que queda a 'desfase' m sobre el suelo.

        Devuelve None si el rayo no toca el suelo dentro del alcance.
        """
        d = np.asarray(d, float) / np.linalg.norm(d)
        if self.plano:
            if d[2] > -1e-6:
                return None
            t = (desfase - o[2]) / d[2]
            return o + t * d if 0 < t <= alcance else None
        paso = max(0.5, self.paso / 2)
        f = lambda t: o[2] + t * d[2] - self.altura(o[0] + t * d[0], o[1] + t * d[1]) - desfase
        t0, f0 = 0.0, f(0.0)
        if f0 <= 0:
            return None
        t = paso
        while t <= alcance:
            f1 = f(t)
            if f1 <= 0:     # cruzó el suelo entre t0 y t: bisección
                for _ in range(25):
                    tm = (t0 + t) / 2
                    if f(tm) > 0:
                        t0 = tm
                    else:
                        t = tm
                return o + t * d
            t0 = t
            t += paso
        return None


def pendiente(terreno, x, y, rumbo, dl=1.0):
    """Ángulo de la pendiente del suelo en la dirección del rumbo [rad]."""
    c, s = math.cos(rumbo), math.sin(rumbo)
    dz = terreno.altura(x + dl * c, y + dl * s) - terreno.altura(x - dl * c, y - dl * s)
    return math.atan2(dz, 2 * dl)
