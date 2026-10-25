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


def test_compensado_sin_actuador_es_el_implicito():
    rng = np.random.default_rng(1)
    a = STA(2, 0.005, 'implicito', k1_0=5.0, adaptativo=False)
    b = STA(2, 0.005, 'compensado', k1_0=5.0, adaptativo=False, tau_actuador=0.0, horizonte=1)
    for s in rng.uniform(-1, 1, (200, 2)):
        assert np.allclose(a(s), b(s))


def banda_con_actuador(esquema, k1, tau=0.04, Ts=0.005, N=2, lam=15.0, T=20.0, sub=10,
                       D_planta=0, D_modelo=0):
    """Canal de actitud: th'' = tau_m + d,  tau_m' = (tau_c - tau_m)/tau,  tau_c = -(lam w + u).
    s = -(w + lam th). El retardo afecta a todo el par, también al término lam w.
    El par comandado llega D_planta muestras tarde. Devuelve max|s| en t > 10 s."""
    c = STA(1, Ts, esquema, k1_0=k1, adaptativo=False, tau_actuador=tau, horizonte=N,
            retardo=D_modelo)
    th, w, tm, t, h, S = 0.1, 0.0, 0.0, 0.0, Ts / sub, []
    cola = [0.0] * D_planta
    for _ in range(int(T / Ts)):
        s = -(w + lam * th)
        cola.append(-(lam * w + c(np.array([s]))[0]))
        tc = cola.pop(0)
        if not np.isfinite(th) or abs(th) > 10:
            return np.inf
        for _ in range(sub):
            th, w = th + h * w, w + h * (tm + 1 + 0.5 * np.sin(2 * np.pi * 0.2 * t))
            tm += h * (tc - tm) / tau
            t += h
        S.append(s)
    return np.abs(S[int(10 / Ts):]).max()


def test_compensado_recupera_la_insensibilidad_a_la_ganancia():
    assert np.isclose(banda_con_actuador('compensado', 10.0), banda_con_actuador('compensado', 120.0), rtol=0.05)
    assert banda_con_actuador('implicito', 120.0) > 3 * banda_con_actuador('implicito', 10.0)


def test_retardo_en_el_modelo_permite_horizonte_corto():
    # Con dos muestras de retardo puro y N = 2, el compensado sin modelo del retardo se
    # degrada con la ganancia; con el retardo en el modelo vuelve a ser insensible.
    sin = [banda_con_actuador('compensado', k, D_planta=2) for k in (10.0, 120.0)]
    con = [banda_con_actuador('compensado', k, D_planta=2, D_modelo=2) for k in (10.0, 120.0)]
    assert sin[1] > 3 * sin[0]
    assert np.isclose(con[0], con[1], rtol=0.05)
