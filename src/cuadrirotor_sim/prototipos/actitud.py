"""Canal de alabeo del Tarot: J th'' = tau_m + dth,  tau_m' = (tau_c - tau_m)/tm.
s = e' + lam e,  tau_c = J (th_r'' + lam e' - u).  Referencia th_r = 0.2 sin(2 pi 0.5 t)."""
import numpy as np, sys
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..'))
from cuadrirotor_sim.sta import SuperTwistingAdaptativo as STA
from eje import STAImpAdapt
J, TAU_MAX = 0.0341, 2.0
def dpar(t): return 0.05 + 0.03*np.sin(2*np.pi*0.3*t)    # par de viento y desbalance [N m]
def corre(ctrl, Ts=0.005, tm_=0.04, lam=15.0, T=30.0, h=0.0005):
    n=int(round(Ts/h)); th,w,tm,t=0.0,0.0,0.0,0.0; E=[];S=[];U=[]
    for k in range(int(T/Ts)):
        r=0.2*np.sin(np.pi*t); rd=0.2*np.pi*np.cos(np.pi*t); rdd=-0.2*np.pi**2*np.sin(np.pi*t)
        e,ed=r-th,rd-w; s=ed+lam*e; u=ctrl(s)
        tc=np.clip(J*(rdd+lam*ed-u),-TAU_MAX,TAU_MAX)
        for _ in range(n):
            th+=h*w; w+=h*(tm+dpar(t))/J; tm = tc if tm_==0 else tm+h*(tc-tm)/tm_; t+=h
        E.append(e);S.append(s);U.append(tc)
        if not np.isfinite(th) or abs(th)>3: return dict(rms=np.inf,sb=np.inf,tv=np.inf,k1=getattr(ctrl,'k1',np.nan))
    m=slice(int(10/Ts),None); E,S,U=map(np.array,(E,S,U))
    k1 = ctrl.k1 if np.isscalar(ctrl.k1) else float(ctrl.k1[0])
    return dict(rms=np.degrees(np.sqrt(np.mean(E[m]**2))),sb=np.percentile(abs(S[m]),95),
                tv=np.sum(abs(np.diff(U[m])))/(T-10),k1=k1)
class Exp:
    def __init__(s,Ts,k1,adapt=False,mu=0.05): s.c=STA(1,Ts,'explicito',k1_0=k1,adaptativo=adapt,mu=mu,k1_max=200,k1_min=1,w1=40)
    @property
    def k1(s): return s.c.k1
    def __call__(s,x): return float(s.c(np.array([x]))[0])
if __name__=='__main__':
    for tm_ in (0.0, 0.04):
        print(f"\n### motor tm = {tm_*1000:.0f} ms   (error en grados RMS | |s| p95 | TV par | k1 final)")
        for Ts in (0.0025,0.005,0.01,0.02):
            fila=[]
            for k1 in (10,40,120):
                a=corre(Exp(Ts,k1),Ts,tm_); b=corre(STAImpAdapt(Ts,'fija',k1_0=k1,estimar=False),Ts,tm_)
                c=corre(STAImpAdapt(Ts,'fija',k1_0=k1,estimar=True),Ts,tm_)
                fila.append(f"k1={k1}: exp {a['rms']:.3f}/{a['tv']:.0f}  imp {b['rms']:.3f}/{b['tv']:.0f}  imp+d^ {c['rms']:.3f}/{c['tv']:.0f}")
            print(f"Ts={Ts*1000:4.1f} ms | "+" || ".join(fila))
