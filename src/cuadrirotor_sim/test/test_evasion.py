"""Pruebas de la evasión reactiva (histograma polar sobre los puntos del LiDAR)."""
import numpy as np

from cuadrirotor_sim.evasion import Evasion
from cuadrirotor_sim.terreno import Terreno


def tronco(x, y, alto=8.0, r=0.6):
    """Puntos de un árbol: cilindro de radio r hasta 'alto' m."""
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    z = np.arange(2.0, alto, 0.5)
    A, Z = np.meshgrid(ang, z)
    return np.stack([x + r * np.cos(A.ravel()), y + r * np.sin(A.ravel()), Z.ravel()], 1)


def volar(ev, c, g, pasos=400, v=0.5):
    """Mueve la consigna hacia el objetivo corregido; regresa la trayectoria."""
    tray = [c.copy()]
    for _ in range(pasos):
        o = ev.corregir(c, g)
        d = o - c
        n = np.linalg.norm(d)
        if n < 1e-6:
            break
        c = c + d / n * min(v, n)
        tray.append(c.copy())
        if np.linalg.norm(c - g) < 0.3:
            break
    return np.array(tray)


def test_camino_libre_no_cambia_el_objetivo():
    ev = Evasion(Terreno(''))
    ev.agregar(tronco(0.0, 15.0), np.zeros(3))
    g = np.array([30.0, 0.0, 5.0])
    assert np.allclose(ev.corregir(np.array([0.0, 0.0, 5.0]), g), g)
    assert not ev.activa


def test_suelo_se_descarta():
    ev = Evasion(Terreno(''))
    ev.agregar(np.array([[5.0, 0.0, 0.3], [6.0, 0.0, 1.0], [7.0, 0.0, 4.0]]), np.zeros(3))
    assert len(ev.puntos) == 1


def test_rodea_un_arbol_en_el_camino():
    ev = Evasion(Terreno(''), radio=3.0)
    ev.agregar(tronco(15.0, 0.3), np.zeros(3))
    tray = volar(ev, np.array([0.0, 0.0, 5.0]), np.array([30.0, 0.0, 5.0]))
    assert np.linalg.norm(tray[-1] - [30.0, 0.0, 5.0]) < 0.5        # llega
    dist = np.min(np.hypot(tray[:, 0] - 15.0, tray[:, 1] - 0.3))
    assert dist > 2.4                                               # sin acercarse al tronco
    assert np.all(np.abs(tray[:, 2] - 5.0) < 1e-9)                   # rodeando, sin subir


def test_cruza_entre_arboles():
    ev = Evasion(Terreno(''), radio=2.5)
    pts = np.vstack([tronco(15.0, y) for y in (-12.0, -4.0, 4.0, 12.0)])   # huecos de 8 m
    ev.agregar(pts, np.zeros(3))
    tray = volar(ev, np.array([0.0, 1.0, 5.0]), np.array([30.0, 1.0, 5.0]), pasos=600)
    assert np.linalg.norm(tray[-1] - [30.0, 1.0, 5.0]) < 0.5
    for y in (-12.0, -4.0, 4.0, 12.0):
        assert np.min(np.hypot(tray[:, 0] - 15.0, tray[:, 1] - y)) > 1.9


def test_sin_salida_sube():
    ev = Evasion(Terreno(''), radio=3.0)
    ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    muro = np.vstack([tronco(8 * np.cos(a), 8 * np.sin(a)) for a in ang])
    ev.agregar(muro, np.zeros(3))
    o = ev.corregir(np.array([0.0, 0.0, 5.0]), np.array([30.0, 0.0, 5.0]))
    assert o[2] > 5.0 and np.allclose(o[:2], 0.0)


def test_no_baja_dentro_de_las_copas():
    ev = Evasion(Terreno(''), radio=3.0)
    ev.agregar(tronco(0.5, 0.0, alto=8.0), np.zeros(3))
    o = ev.corregir(np.array([0.0, 0.0, 20.0]), np.array([0.0, 0.0, 4.0]))
    assert o[2] >= 7.5 + 3.0 - ev.voxel     # el vóxel de 1 m redondea la copa


def test_olvida_lo_lejano():
    ev = Evasion(Terreno(''), memoria=60.0)
    ev.agregar(tronco(10.0, 0.0), np.zeros(3))
    n = len(ev.puntos)
    assert n > 0
    ev.agregar(np.zeros((0, 3)), np.array([200.0, 0.0, 0.0]))
    assert len(ev.puntos) == 0
