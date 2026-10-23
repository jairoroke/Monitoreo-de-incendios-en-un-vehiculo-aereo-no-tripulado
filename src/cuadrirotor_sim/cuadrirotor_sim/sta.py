"""Algoritmo super-twisting con ganancia adaptativa y tres realizaciones discretas.

Sistema en tiempo continuo, con s la variable deslizante y u la entrada:

    s' = u + d(t),      u = -k1 |s|^(1/2) sgn(s) + nu,     nu' = -k2 sgn(s)

Ganancia adaptativa (Shtessel, Taleb y Plestan, 2012):

    k1' = w1 sqrt(g1/2) sgn(|s| - mu)   si k1 > k1_min,   k1' = eta  en otro caso
    k2  = 2 eps k1

Realizaciones discretas con periodo Ts (a = s_k + Ts nu_k):

  explicito       u_k = -k1 |s_k|^(1/2) sgn(s_k) + nu_k,   nu_{k+1} = nu_k - Ts k2 sgn(s_k)

  semi_implicito  el término discontinuo se evalúa en el paso siguiente pero la
                  ganancia |s|^(1/2) en el paso actual:
                      s~ + (Ts k1 |s_k|^(1/2) + Ts^2 k2) xi = a,   xi en Sgn(s~)

  implicito       (Brogliato, Polyakov y Efimov, 2020):
                      s~ + Ts k1 |s~|^(1/2) xi + Ts^2 k2 xi = a,   xi en Sgn(s~)
                  si |a| <= Ts^2 k2:  s~ = 0,  xi = a / (Ts^2 k2)
                  en otro caso:       s~ = sgn(a) w^2,
                                      w = (-Ts k1 + sqrt(Ts^2 k1^2 + 4(|a| - Ts^2 k2))) / 2

En los dos esquemas implícitos s~ es la predicción nominal de s_{k+1},
nu_{k+1} = nu_k - Ts k2 xi y u_k = (s~ - s_k) / Ts.

La predicción nominal supone d = 0, de modo que con perturbación la banda en
régimen es |s| ~ Ts |d|, independiente de la ganancia. Con
estimar_perturbacion=True se usa d^_k = (s_k - s_{k-1}) / Ts - u_{k-1}
(Acary, Brogliato y Orlov, 2012): a = s_k + Ts (nu_k + d^_k) y
u_k = (s~ - s_k) / Ts - d^_k, lo que reduce la banda a ~ Ts^2 max|d'|.
Todo opera componente a componente sobre arreglos de numpy.
"""
import numpy as np

ESQUEMAS = ('explicito', 'semi_implicito', 'implicito')


class SuperTwistingAdaptativo:

    def __init__(self, n, Ts, esquema='implicito', k1_0=1.0, eps=1.0, w1=2.0, g1=1.0,
                 mu=0.02, k1_min=0.1, eta=0.1, k1_max=30.0, adaptativo=True,
                 estimar_perturbacion=False):
        if esquema not in ESQUEMAS:
            raise ValueError(f'esquema debe ser uno de {ESQUEMAS}, no {esquema!r}')
        self.Ts, self.esquema = Ts, esquema
        self.eps, self.mu = eps, mu
        self.tasa = w1 * np.sqrt(g1 / 2.0)
        self.k1_min, self.eta, self.k1_max = k1_min, eta, k1_max
        self.adaptativo = adaptativo
        self.estimar = estimar_perturbacion and esquema != 'explicito'
        self.s_prev = self.u_prev = None
        self.k1 = np.full(n, float(k1_0))
        self.nu = np.zeros(n)

    @property
    def k2(self):
        return 2.0 * self.eps * self.k1

    def __call__(self, s):
        """Devuelve u_k para la medición s_k y avanza el estado interno."""
        s = np.asarray(s, dtype=float)
        Ts, k1, k2 = self.Ts, self.k1, self.k2

        if self.esquema == 'explicito':
            sg = np.sign(s)
            u = -k1 * np.sqrt(np.abs(s)) * sg + self.nu
            self.nu = self.nu - Ts * k2 * sg
        else:
            d_est = 0.0
            if self.estimar and self.s_prev is not None:
                d_est = (s - self.s_prev) / Ts - self.u_prev
            a = s + Ts * (self.nu + d_est)
            if self.esquema == 'implicito':
                s_pred, xi = self.resolver_implicito(a, Ts * k1, Ts**2 * k2)
            else:
                c = Ts * k1 * np.sqrt(np.abs(s)) + Ts**2 * k2
                dentro = np.abs(a) <= c
                xi = np.where(dentro, a / np.where(dentro, c, 1.0), np.sign(a))
                s_pred = np.where(dentro, 0.0, a - c * np.sign(a))
            self.nu = self.nu - Ts * k2 * xi
            u = (s_pred - s) / Ts - d_est
            self.s_prev, self.u_prev = s, u

        if self.adaptativo:
            dk = np.where(self.k1 > self.k1_min, self.tasa * np.sign(np.abs(s) - self.mu), self.eta)
            self.k1 = np.clip(self.k1 + Ts * dk, self.k1_min, self.k1_max)
        return u

    @staticmethod
    def resolver_implicito(a, b, c):
        """Solución única de  x + b |x|^(1/2) xi + c xi = a,  xi en Sgn(x),  b, c > 0."""
        abs_a = np.abs(a)
        dentro = abs_a <= c
        w = (-b + np.sqrt(b**2 + 4.0 * np.maximum(abs_a - c, 0.0))) / 2.0
        x = np.where(dentro, 0.0, np.sign(a) * w**2)
        xi = np.where(dentro, a / np.where(dentro, c, 1.0), np.sign(a))
        return x, xi
