"""STA implícito con la dinámica de primer orden del actuador dentro de la inclusión.
u_m' = (u - u_m)/tm (modelo del motor; u_m se integra en el controlador).
Con u_k retenido durante el horizonte H = N Ts, el promedio de u_m es
  g u_k + (1-g) u_m,k,   g = 1 - (tm/H)(1 - exp(-H/tm)).
Paso implícito sobre el horizonte H:  x + H k1 |x|^(1/2) xi + H^2 k2 xi = s_k + H nu_k,
y u_k = ((x - s_k)/H - (1-g) u_m,k) / g."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cuadrirotor_sim.sta import SuperTwistingAdaptativo as STA
class STAImpComp:
    def __init__(s, Ts, tm, N=1, k1_0=10.0, eps=1.0, ley='fija', mu=0.05, k1_min=1.0, k1_max=200.0, w=40.0, eta=1.0):
        s.Ts, s.H = Ts, N*Ts
        s.al = np.exp(-Ts/tm) if tm > 0 else 0.0
        s.g = 1 - (tm/s.H)*(1-np.exp(-s.H/tm)) if tm > 0 else 1.0
        s.k1, s.eps, s.ley, s.mu, s.k1_min, s.k1_max, s.w, s.eta = k1_0, eps, ley, mu, k1_min, k1_max, w, eta
        s.nu = 0.0; s.um = 0.0
    def __call__(s, sk):
        H, k1 = s.H, s.k1; k2 = 2*s.eps*k1
        x, xi = STA.resolver_implicito(np.array([sk + H*s.nu]), H*k1, H*H*k2)
        x, xi = float(x[0]), float(xi[0])
        s.nu -= s.Ts*k2*xi
        u = ((x - sk)/H - (1-s.g)*s.um)/s.g
        s.um = s.al*s.um + (1-s.al)*u
        if s.ley == 'mu_fijo':
            dk = s.w*np.sign(abs(sk)-s.mu) if s.k1 > s.k1_min else s.eta
            s.k1 = float(np.clip(s.k1 + s.Ts*dk, s.k1_min, s.k1_max))
        return u
