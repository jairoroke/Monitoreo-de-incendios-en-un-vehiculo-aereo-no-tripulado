import math

import numpy as np

from cuadrirotor_sim.dinamica import Mezclador, Referencia, quat_a_rot, rot_a_euler


def test_mezclador_invierte_la_asignacion():
    m = Mezclador.configuracion('mas', 0.225, 8.54858e-06, 0.016, 1000.0)
    T, tau = 15.0, np.array([0.3, -0.2, 0.05])
    f = m.fuerzas(T, tau)
    l, km = 0.225, m.km
    assert math.isclose(f.sum(), T)
    assert math.isclose(l * (f[1] - f[3]), tau[0])
    assert math.isclose(l * (f[2] - f[0]), tau[1])
    assert math.isclose(km * (-f[0] + f[1] - f[2] + f[3]), tau[2])


def test_mezclador_x_y_geometria_irregular():
    # Disposición del Tarot 650 medida en el CAD (no es exactamente cuadrada).
    pos = [(0.2198, -0.212), (0.2198, 0.212), (-0.2198, 0.212), (-0.2198, -0.212)]
    m = Mezclador(pos, [1, -1, 1, -1], 4.24e-05, 0.016, 680.0)
    T, tau = 21.6, np.array([0.4, -0.3, 0.08])
    f = m.fuerzas(T, tau)
    assert np.allclose(m.B @ f, [T, *tau])
    # Sin pares, el empuje se reparte por igual.
    assert np.allclose(m.fuerzas(T, np.zeros(3)), T / 4)


def test_velocidad_de_vuelo_estacionario():
    m = Mezclador.configuracion('x', 0.3054, 8.54858e-06, 0.016, 1000.0)
    w = m.velocidades(1.5 * 9.81, np.zeros(3))
    assert np.allclose(w, math.sqrt(1.5 * 9.81 / 4 / 8.54858e-06))


def test_euler_de_cuaternion():
    phi, th, psi = 0.1, -0.2, 0.3
    cy, sy = math.cos(psi / 2), math.sin(psi / 2)
    cp, sp = math.cos(th / 2), math.sin(th / 2)
    cr, sr = math.cos(phi / 2), math.sin(phi / 2)
    q = (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
         cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)
    assert np.allclose(rot_a_euler(quat_a_rot(*q)), (phi, th, psi))


def test_referencia_continua():
    ref = Referencia([0, 0, 0.05], mision='orbita')
    ts = np.arange(0, 40, 0.001)
    P = np.array([ref(t)[0] for t in ts])
    V = np.array([ref(t)[1] for t in ts])
    psi = np.array([ref(t)[3] for t in ts])
    assert np.abs(np.diff(P, axis=0)).max() < 0.01
    assert np.abs(np.diff(V, axis=0)).max() < 0.01
    assert np.abs(np.diff(psi)).max() < 0.01
    # La velocidad de la referencia coincide con la derivada numérica de la posición.
    assert np.allclose(np.gradient(P, ts, axis=0)[1:-1], V[1:-1], atol=1e-3)
