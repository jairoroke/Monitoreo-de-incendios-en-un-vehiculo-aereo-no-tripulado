import numpy as np
import pytest

from cuadrirotor_sim.sta import ESQUEMAS, SuperTwistingAdaptativo as STA


def test_solucion_implicita_cumple_la_inclusion():
    rng = np.random.default_rng(0)
    a = rng.uniform(-2, 2, 1000)
    b, c = 0.03, 4e-4
    x, xi = STA.resolver_implicito(a, b, c)
    assert np.allclose(x + b * np.sqrt(np.abs(x)) * xi + c * xi, a)
    assert np.all(np.abs(xi) <= 1 + 1e-12)
    assert np.allclose(xi[x != 0], np.sign(x[x != 0]))


def banda(esquema, Ts, k1, estimar=False, T=20.0, sub=20):
    """Doble integrador s' = u + d con d = 1 + 0.5 sin(2 pi 0.2 t); devuelve max|s| en t > 10 s."""
    c = STA(1, Ts, esquema, k1_0=k1, adaptativo=False, estimar_perturbacion=estimar)
    s, t, h, S = np.array([1.0]), 0.0, Ts / sub, []
    for _ in range(int(T / Ts)):
        u = c(s)
        for _ in range(sub):
            s = s + h * (u + 1 + 0.5 * np.sin(2 * np.pi * 0.2 * t))
            t += h
        S.append(s[0])
    return np.abs(S[int(10 / Ts):]).max()


@pytest.mark.parametrize('esquema', ESQUEMAS)
def test_converge_con_perturbacion(esquema):
    assert banda(esquema, 0.005, 3.0) < 0.01


def test_implicito_insensible_a_la_ganancia():
    assert np.isclose(banda('implicito', 0.005, 3.0), banda('implicito', 0.005, 30.0), rtol=0.05)
    assert banda('explicito', 0.005, 30.0) > 10 * banda('explicito', 0.005, 3.0)


def test_estimacion_reduce_la_banda_a_orden_ts_cuadrado():
    b1 = banda('implicito', 0.01, 3.0, estimar=True)
    b2 = banda('implicito', 0.005, 3.0, estimar=True)
    assert b1 < 1e-4 and 3.5 < b1 / b2 < 4.5
