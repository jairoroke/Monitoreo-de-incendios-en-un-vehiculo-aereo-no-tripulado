"""Prototipo: un eje horizontal del Tarot 650 con el lazo de actitud real y STA implícito adaptativo.

p'' = g th + d,  th_r = a_c / g,  J th'' = tau_m,  tau_m' = (tau_c - tau_m)/tm,
tau_c = -kr (th - th_r) - kw th'  (PD de config/vehiculo_tarot650.yaml).
La velocidad medida es el promedio de las últimas 10 muestras a 1 kHz (como Gazebo).
"""
import numpy as np, sys
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..'))
from cuadrirotor_sim.sta import SuperTwistingAdaptativo as STA

G, J, KR, KW, TM = 9.81, 0.0341, 3.4, 0.55, 0.04
LAM = 1.5

def viento(t, rafagas=False):
    if t < 15: return 0.0
    d = (1.0 + 0.5*np.sin(2*np.pi*0.2*t)) / 2.2
    if rafagas: d += 0.6/2.2 * (np.sin(2*np.pi*1.3*t) > 0.7)
    return d

class STAImpAdapt:
    """STA implícito con d^ y ley de adaptación intercambiable."""
    def __init__(self, Ts, ley, mu=0.3, k1_0=1.0, k1_min=0.5, k1_max=8.0, w=np.sqrt(2.0),
                 eta=0.1, c=3.0, mu0=0.005, tf=1.0, estimar=True):
        self.Ts, self.ley, self.mu = Ts, ley, mu
        self.k1, self.k1_min, self.k1_max, self.w, self.eta = k1_0, k1_min, k1_max, w, eta
        self.c, self.mu0, self.alfa = c, mu0, Ts/tf
        self.estimar = estimar
        self.nu = 0.0; self.sp = self.up = None; self.dp = 0.0; self.sf = 0.0; self.xi = 1.0; self.muk = mu
    def __call__(self, s):
        Ts, k1 = self.Ts, self.k1; k2 = 2*k1
        dh = 0.0
        if self.estimar and self.sp is not None:
            dh = (s - self.sp)/Ts - self.up
        a = s + Ts*(self.nu + dh)
        x, xi = STA.resolver_implicito(np.array([a]), Ts*k1, Ts**2*k2)
        x, xi = float(x[0]), float(xi[0])
        self.nu -= Ts*k2*xi
        u = (x - s)/Ts - dh
        # señal de adaptación: >0 sube k1, <0 baja
        if self.ley == 'mu_fijo':
            sig = abs(s) - self.mu
        elif self.ley == 'mu_banda':        # umbral deducido de la banda del implícito con d^
            self.muk = self.c*Ts*abs(dh - self.dp) + self.mu0
            sig = abs(s) - self.muk
        elif self.ley == 'xi':              # |xi|<1: deslizamiento discreto alcanzado
            sig = abs(xi) - 0.999
        elif self.ley == 'filtrada':        # media móvil de s: el ciclo límite es de media cero
            self.sf += self.alfa*(s - self.sf)
            sig = abs(self.sf) - self.mu
        elif self.ley == 'fija':
            sig = 0.0
        self.dp = dh
        if self.ley != 'fija':
            dk = self.w*np.sign(sig) if self.k1 > self.k1_min else self.eta
            self.k1 = float(np.clip(self.k1 + Ts*dk, self.k1_min, self.k1_max))
        self.sp, self.up = s, u
        return u

def corre(ctrl, Ts=0.005, T=60.0, h=0.001, rafagas=False, p0=1.0):
    n_sub = int(round(Ts/h))
    p, v, th, w, tm = p0, 0.0, 0.0, 0.0, 0.0
    vbuf = [0.0]*10; th_r = 0.0
    P, K, A, S = [], [], [], []
    t = 0.0
    for k in range(int(T/Ts)):
        vm = np.mean(vbuf)
        e, ed = -p, -vm
        s = ed + LAM*e
        u = ctrl(s)
        ac = np.clip(LAM*ed - u, -6, 6)
        th_r = np.arctan(ac/G)
        for _ in range(n_sub):
            tc = -KR*(th - th_r) - KW*w
            a = G*np.tan(th) + viento(t, rafagas)
            p += h*v; v += h*a; th += h*w; w += h*tm/J; tm += h*(tc - tm)/TM
            vbuf = vbuf[1:] + [v]
            t += h
        P.append(p); K.append(ctrl.k1); A.append(ac); S.append(s)
        if not np.isfinite(p) or abs(p) > 50: return dict(rms=np.inf, k1f=ctrl.k1, k1max=max(K), tv=np.inf, sb=np.inf)
    P, K, A, S = map(np.array, (P, K, A, S)); m = np.arange(len(P))*Ts >= 20
    return dict(rms=np.sqrt(np.mean(P[m]**2))*100, k1f=K[-1], k1max=K.max(),
                tv=np.sum(np.abs(np.diff(A[m])))/(T-20), sb=np.percentile(np.abs(S[m]), 95))

if __name__ == '__main__':
    casos = [('fija k1=1', dict(ley='fija', k1_0=1.0)),
             ('fija k1=3', dict(ley='fija', k1_0=3.0)),
             ('fija k1=8', dict(ley='fija', k1_0=8.0)),
             ('mu fijo 0.30 (actual)', dict(ley='mu_fijo', mu=0.3)),
             ('mu fijo 0.05', dict(ley='mu_fijo', mu=0.05)),
             ('mu de la banda', dict(ley='mu_banda')),
             ('criterio xi', dict(ley='xi')),
             ('s filtrada mu=0.02', dict(ley='filtrada', mu=0.02)),
             ('s filtrada mu=0.005', dict(ley='filtrada', mu=0.005))]
    for raf in (False, True):
        print(f"\n== viento {'con ráfagas' if raf else 'senoidal'} | Ts=5 ms | k1 en [0.5, 8] ==")
        print(f"{'ley':24s} {'RMS cm':>8s} {'k1 fin':>7s} {'k1 máx':>7s} {'|s|p95':>7s} {'TV a_c':>8s}")
        for nom, kw in casos:
            r = corre(STAImpAdapt(0.005, **kw), rafagas=raf)
            print(f"{nom:24s} {r['rms']:8.2f} {r['k1f']:7.2f} {r['k1max']:7.2f} {r['sb']:7.3f} {r['tv']:8.1f}")
