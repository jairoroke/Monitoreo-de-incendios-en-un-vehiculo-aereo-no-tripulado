"""Utilidades del modelo Newton-Euler: rotaciones, mezclador y referencias.

No depende de ROS, de modo que puede probarse y reutilizarse por separado.
"""
import math

import numpy as np


def quat_a_rot(qx, qy, qz, qw):
    """Matriz de rotación cuerpo -> mundo a partir de un cuaternión."""
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def rot_a_euler(R):
    """Ángulos (phi, theta, psi) en convención ZYX."""
    phi = math.atan2(R[2, 1], R[2, 2])
    theta = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    psi = math.atan2(R[1, 0], R[0, 0])
    return phi, theta, psi


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def envolver(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Mezclador:
    """Convierte (empuje, pares) en velocidades de rotor para cualquier disposición.

    El rotor i está en (x_i, y_i) del marco del cuerpo y gira en sentido s_i
    (+1 ccw, -1 cw). Con f_i = kf w_i^2, la matriz de asignación es
        T  = sum f_i
        tx = sum  y_i f_i
        ty = sum -x_i f_i
        tz = sum -s_i km f_i
    y las fuerzas se obtienen con su inversa.
    """

    def __init__(self, posiciones, sentidos, kf, km, w_max):
        self.kf, self.km, self.w_max = kf, km, w_max
        self.f_max = kf * w_max ** 2
        xy = np.asarray(posiciones, dtype=float)
        s = np.asarray(sentidos, dtype=float)
        self.B = np.vstack([np.ones(len(s)), xy[:, 1], -xy[:, 0], -s * km])
        self.B_inv = np.linalg.pinv(self.B)

    @classmethod
    def configuracion(cls, tipo, l, kf, km, w_max):
        """Rotores a distancia l: tipo 'mas' (rotor 0 en +x) o 'x' (rotor 0 adelante-derecha)."""
        if tipo == 'mas':
            pos = [(l, 0.0), (0.0, l), (-l, 0.0), (0.0, -l)]
        elif tipo == 'x':
            c = l / math.sqrt(2.0)
            pos = [(c, -c), (c, c), (-c, c), (-c, -c)]
        else:
            raise ValueError(f"configuración desconocida: {tipo!r}")
        return cls(pos, [1, -1, 1, -1], kf, km, w_max)

    def fuerzas(self, T, tau):
        return self.B_inv @ np.array([T, *tau])

    def velocidades(self, T, tau):
        f = np.clip(self.fuerzas(T, tau), 0.0, self.f_max)
        return np.sqrt(f / self.kf)

    def realizado(self, w):
        """Empuje y pares que producen las velocidades de rotor w (ya saturadas)."""
        return self.B @ (self.kf * np.asarray(w) ** 2)


def quintica(s):
    """Perfil suave 0 -> 1 y sus dos derivadas respecto a s en [0, 1]."""
    s = min(max(s, 0.0), 1.0)
    return (10 * s**3 - 15 * s**4 + 6 * s**5,
            30 * s**2 - 60 * s**3 + 30 * s**4,
            60 * s - 180 * s**2 + 120 * s**3)


class Referencia:
    """Genera pd(t), vd(t), ad(t) y psi_d(t).

    mision = 'hover':  despegue a (x0, y0, h) y vuelo estacionario.
    mision = 'orbita': despegue, traslado a (xf + R, yf, h) y órbita circular
                       alrededor del foco con la cámara apuntando hacia él.
    """

    def __init__(self, p0, mision='hover', altura=3.0, foco=(8.0, 5.0), radio=4.0,
                 omega=0.3, t_ascenso=5.0, t_traslado=8.0, t_rampa=4.0):
        self.p0 = np.array(p0, dtype=float)
        self.mision, self.h = mision, altura
        self.xf, self.yf = foco
        self.R, self.w = radio, omega
        self.Ta, self.Tt, self.Tr = t_ascenso, t_traslado, t_rampa

    def __call__(self, t):
        pa = np.array([self.p0[0], self.p0[1], self.h])
        if t < self.Ta or self.mision == 'hover':
            s, ds, dds = quintica(t / self.Ta)
            p = self.p0 + (pa - self.p0) * s
            v = (pa - self.p0) * ds / self.Ta
            a = (pa - self.p0) * dds / self.Ta**2
            return p, v, a, 0.0

        t1 = t - self.Ta
        pb = np.array([self.xf + self.R, self.yf, self.h])
        if t1 < self.Tt:
            s, ds, dds = quintica(t1 / self.Tt)
            p = pa + (pb - pa) * s
            v = (pb - pa) * ds / self.Tt
            a = (pb - pa) * dds / self.Tt**2
            return p, v, a, math.pi * s

        # Órbita: la velocidad angular sube de 0 a omega durante t_rampa.
        tau = t1 - self.Tt
        if tau < self.Tr:
            th, dth, ddth = self.w * tau**2 / (2 * self.Tr), self.w * tau / self.Tr, self.w / self.Tr
        else:
            th, dth, ddth = self.w * (tau - self.Tr / 2), self.w, 0.0
        c, s = math.cos(th), math.sin(th)
        p = np.array([self.xf + self.R * c, self.yf + self.R * s, self.h])
        v = np.array([-self.R * s * dth, self.R * c * dth, 0.0])
        a = np.array([-self.R * (c * dth**2 + s * ddth), self.R * (-s * dth**2 + c * ddth), 0.0])
        return p, v, a, th + math.pi
