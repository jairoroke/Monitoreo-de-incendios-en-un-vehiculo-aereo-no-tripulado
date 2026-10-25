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

  compensado      implícito con la dinámica de primer orden del actuador,
                  u_m' = (u - u_m) / tau, dentro de la inclusión. Con u_k retenido
                  durante el horizonte H = N Ts, el promedio de u_m es
                  g u_k + (1 - g) u_m,k  con  g = 1 - (tau/H)(1 - exp(-H/tau)),
                  y la inclusión tiene la misma forma con H en lugar de Ts:
                      s~ + H k1 |s~|^(1/2) xi + H^2 k2 xi = s_k + H nu_k,
                  u_k = ((s~ - s_k)/H - (1 - g) u_m,k) / g.  u_m se integra con el
                  modelo (u_m,k+1 = a u_m,k + (1 - a) u_k, a = exp(-Ts/tau)).
                  Con retardo = D muestras, u_k actúa hasta k + D: los D comandos ya
                  enviados se propagan con el mismo modelo (avanzan s en ds y dejan
                  u_m en u_m,k+D), la inclusión se plantea sobre H_t = D Ts + H
                  y u_k = ((s~ - s_k - ds)/H - (1 - g) u_m,k+D) / g.
                  Con tau = 0, N = 1 y D = 0 coincide con el implícito.
                  tau puede darse por componente: en guiñada el par de reacción del
                  rotor (I_r dω/dt) llega sin el retraso del motor y anula el polo.

La predicción nominal supone d = 0, de modo que con perturbación la banda en
régimen es |s| ~ Ts |d|, independiente de la ganancia. Con
estimar_perturbacion=True se usa d^_k = (s_k - s_{k-1}) / Ts - u_{k-1}
(Acary, Brogliato y Orlov, 2012): a = s_k + Ts (nu_k + d^_k) y
u_k = (s~ - s_k) / Ts - d^_k, lo que reduce la banda a ~ Ts^2 max|d'|.
Todo opera componente a componente sobre arreglos de numpy.
"""
import numpy as np

ESQUEMAS = ('explicito', 'semi_implicito', 'implicito', 'compensado')


class SuperTwistingAdaptativo:

    def __init__(self, n, Ts, esquema='implicito', k1_0=1.0, eps=1.0, w1=2.0, g1=1.0,
                 mu=0.02, k1_min=0.1, eta=0.1, k1_max=30.0, adaptativo=True,
                 estimar_perturbacion=False, tau_actuador=0.0, horizonte=1, retardo=0):
        if esquema not in ESQUEMAS:
            raise ValueError(f'esquema debe ser uno de {ESQUEMAS}, no {esquema!r}')
        self.Ts, self.esquema = Ts, esquema
        self.eps, self.mu = eps, mu
        self.tasa = w1 * np.sqrt(g1 / 2.0)
        self.k1_min, self.eta, self.k1_max = k1_min, eta, k1_max
        self.adaptativo = adaptativo
        self.estimar = estimar_perturbacion and esquema in ('semi_implicito', 'implicito')
        self.H = horizonte * Ts
        # tau_actuador puede ser distinto por componente; tau = 0 es actuador instantáneo.
        tau = np.broadcast_to(np.asarray(tau_actuador, dtype=float), (n,))
        con = tau > 0
        tau_s = np.where(con, tau, 1.0)
        self.g = np.where(con, 1.0 - (tau_s / self.H) * (1.0 - np.exp(-self.H / tau_s)), 1.0)
        self.a_act = np.where(con, np.exp(-Ts / tau_s), 0.0)
        # Promedio de u_m durante un periodo: u_q + (u_m - u_q) b1.
        self.b1 = np.where(con, (tau_s / Ts) * (1.0 - self.a_act), 0.0)
        self.u_m = self.u_ult = np.zeros(n)
        self.cola = [np.zeros(n) for _ in range(retardo)]   # comandos enviados que aún no actúan
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

        if self.esquema == 'compensado':
            # Los D comandos ya enviados avanzan s y u_m antes de que actúe u_k.
            a, um, ds = self.a_act, self.u_m, 0.0
            for uq in self.cola:
                ds = ds + Ts * (uq + (um - uq) * self.b1)
                um = a * um + (1.0 - a) * uq
            H, Ht = self.H, self.H + len(self.cola) * Ts
            s_pred, xi = self.resolver_implicito(s + Ht * self.nu, Ht * k1, Ht**2 * k2)
            self.nu = self.nu - Ts * k2 * xi
            u = ((s_pred - s - ds) / H - (1.0 - self.g) * um) / self.g
            self.cola.append(u)
            self.u_m = a * self.u_m + (1.0 - a) * self.cola.pop(0)
            self.u_ult = u
        elif self.esquema == 'explicito':
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

    def aplicado(self, u):
        """Corrige el modelo del actuador con la u que de verdad se aplicó (saturación)."""
        if self.esquema != 'compensado':
            return
        u = np.asarray(u, dtype=float)
        if self.cola:
            self.cola[-1] = u
        else:
            self.u_m = self.u_m + (1.0 - self.a_act) * (u - self.u_ult)

    @staticmethod
    def resolver_implicito(a, b, c):
        """Solución única de  x + b |x|^(1/2) xi + c xi = a,  xi en Sgn(x),  b, c > 0."""
        abs_a = np.abs(a)
        dentro = abs_a <= c
        w = (-b + np.sqrt(b**2 + 4.0 * np.maximum(abs_a - c, 0.0))) / 2.0
        x = np.where(dentro, 0.0, np.sign(a) * w**2)
        xi = np.where(dentro, a / np.where(dentro, c, 1.0), np.sign(a))
        return x, xi
