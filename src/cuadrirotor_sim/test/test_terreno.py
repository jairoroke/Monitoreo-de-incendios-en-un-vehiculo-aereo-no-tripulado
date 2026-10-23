"""Pruebas del DEM, la ruta de barrido y el mapa de cobertura."""
import math
import os

import numpy as np
import pytest

from cuadrirotor_sim.cobertura import Cobertura, cortar_rayos, rayos_camara, ruta_barrido
from cuadrirotor_sim.terreno import Terreno

DEM = os.path.join(os.path.dirname(__file__), '..', 'models', 'terreno_bosque_grande', 'altura.npz')


@pytest.fixture
def rampa(tmp_path):
    """Plano inclinado z = 0.1 x + 0.05 y en una malla de 2 m."""
    ejes = np.arange(-50.0, 50.1, 2.0)
    x, y = np.meshgrid(ejes, ejes)
    ruta = tmp_path / 'rampa.npz'
    np.savez(ruta, x0=-50.0, y0=-50.0, paso=2.0, h=0.1 * x + 0.05 * y)
    return Terreno(str(ruta))


def test_plano_es_cero():
    t = Terreno('')
    assert t.altura(3.0, -7.0) == 0.0
    assert np.all(t.altura(np.array([1.0, 2.0]), np.array([0.0, 5.0])) == 0)


def test_interpola_un_plano_exacto(rampa):
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-45, 45, 200), rng.uniform(-45, 45, 200)
    assert np.allclose(rampa.altura(x, y), 0.1 * x + 0.05 * y, atol=1e-9)


def test_rayo_toca_la_rampa(rampa):
    o = np.array([-20.0, 0.0, 30.0])
    d = np.array([1.0, 0.0, -1.0])
    p = rampa.cortar_rayo(o, d, alcance=200.0, desfase=1.1)
    assert p is not None
    assert p[2] == pytest.approx(0.1 * p[0] + 0.05 * p[1] + 1.1, abs=1e-3)
    assert rampa.cortar_rayo(o, np.array([1.0, 0.0, 0.2])) is None   # hacia arriba


def test_rayo_en_el_plano():
    p = Terreno('').cortar_rayo(np.array([0.0, 0.0, 10.0]), np.array([1.0, 0.0, -1.0]), desfase=1.1)
    assert p == pytest.approx([8.9, 0.0, 1.1])


def test_cortar_rayos_coincide_con_cortar_rayo(rampa):
    o = np.array([0.0, 0.0, 25.0])
    R = np.array([[math.cos(0.5), 0, math.sin(0.5)], [0, 1, 0], [-math.sin(0.5), 0, math.cos(0.5)]])
    dirs = rayos_camara(R, 300.0, 640, 480, 6, 4)
    puntos = cortar_rayos(rampa, o, dirs, 150.0, paso=0.25)
    for d, p in zip(dirs, puntos):
        exacto = rampa.cortar_rayo(o, d, 150.0)
        if exacto is None:
            continue
        assert np.linalg.norm(p - exacto) < 0.5


def test_ruta_barrido_cubre_la_zona():
    zona = (-210.0, 210.0, -210.0, 210.0)
    ruta = ruta_barrido(zona, 70.0, (-232.0, -232.0))
    assert tuple(ruta[0]) == pytest.approx((-175.0, -175.0))   # esquina más cercana
    ys = sorted(set(ruta[:, 1]))
    assert len(ys) == 6 and np.allclose(np.diff(ys), 70.0)
    assert np.all(np.abs(ruta) <= 210.0)
    # Franjas alternadas: cada par de vértices comparte y, y cambia de lado.
    assert np.all(ruta[0::2, 1] == ruta[1::2, 1])


def test_cobertura_marca_celdas():
    c = Cobertura((0.0, 100.0, 0.0, 50.0), 10.0)
    c.marcar(np.array([[5.0, 5.0, 0.0], [95.0, 45.0, 0.0], [np.nan] * 3, [500.0, 5.0, 0.0]]))
    assert c.vista.shape == (5, 10)
    assert c.vista[0, 0] and c.vista[4, 9]
    assert c.fraccion() == pytest.approx(2 / 50)


@pytest.mark.skipif(not os.path.exists(DEM), reason='falta el DEM del bosque grande')
def test_dem_bosque_grande():
    t = Terreno(DEM)
    assert abs(t.altura(-232.0, -232.0)) < 0.05      # base plana a nivel 0
    assert t.altura(400.0, 400.0) > 60.0             # montañas fuera del valle
    assert np.all(np.abs(t.altura(np.linspace(-200, 200, 50), np.zeros(50))) < 60.0)


def test_ruta_serpentina_avanza_hacia_el_frente():
    from cuadrirotor_sim.cobertura import ruta_serpentina
    zona = (-210.0, 210.0, -210.0, 210.0)
    for rumbo, eje, signo in ((0.0, 0, 1), (math.pi / 2, 1, 1), (math.pi, 0, -1), (-1.4, 1, -1)):
        ruta = ruta_serpentina(zona, 70.0, (-232.0, -232.0), rumbo, paso=5.0)
        assert np.all(np.abs(ruta) <= 210.0 + 1e-9)
        assert np.all(np.diff(signo * ruta[:, eje]) >= -1e-9)        # nunca retrocede
        d = np.linalg.norm(np.diff(ruta, axis=0), axis=1)
        assert d.max() < 5.5 and d.min() > 0.5                      # muestreo parejo
        lat = ruta[:, 1 - eje]
        assert lat.max() - lat.min() > 340                          # barre de lado a lado
    ruta = ruta_serpentina(zona, 70.0, (-232.0, -232.0), 0.0)
    assert tuple(ruta[0]) == pytest.approx((-175.0, -175.0))
    assert ruta[-1, 0] == pytest.approx(175.0)
    # Sin esquinas: el rumbo cambia poco entre puntos (radio de vuelta 35 m).
    rumbos = np.unwrap(np.arctan2(*np.diff(ruta, axis=0).T[::-1]))
    assert np.abs(np.diff(rumbos)).max() < 0.2
