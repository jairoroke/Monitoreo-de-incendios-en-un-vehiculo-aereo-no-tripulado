#!/usr/bin/env python3
"""Resume la campaña de actitud (scripts/campana_actitud.sh).  Uso: resumen_actitud.py <directorio>

Por corrida, con viento (t > 20 s): error de actitud RMS en alabeo/cabeceo, |s_R| p95,
error de posición RMS, chattering (media de |Δω| de los rotores por muestra),
fracción de muestras con algún rotor saturado y k1 final de actitud.
"""
import glob, os, sys
import numpy as np
import pandas as pd

print(f"{'corrida':16s} {'eR° RMS':>8s} {'|sR| p95':>9s} {'pos cm':>7s} {'chatt.':>7s} {'sat %':>6s} {'k1R fin':>8s}")
for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.csv'))):
    d = pd.read_csv(f)
    b = (d.t > 20).values
    if b.sum() < 100:
        print(f"{os.path.basename(f)[:-4]:16s} corrida incompleta"); continue
    eR = np.degrees(np.sqrt((d.loc[b, ['eRx', 'eRy']] ** 2).values.mean()))
    e = np.sqrt((d.x - d.xd) ** 2 + (d.y - d.yd) ** 2 + (d.z - d.zd) ** 2)[b]
    w = d[['w0', 'w1', 'w2', 'w3']].values
    ch = np.abs(np.diff(w, axis=0))[b[1:]].mean()
    sat = 100 * ((w[b] < 1) | (w[b] > 679)).any(axis=1).mean()
    sR = np.percentile(np.abs(d.loc[b, ['sRx', 'sRy']].values), 95) if 'sRx' in d else np.nan
    k1 = d.k1Rx.iloc[-1] if 'k1Rx' in d else np.nan
    print(f"{os.path.basename(f)[:-4]:16s} {eR:8.3f} {sR:9.3f} {np.sqrt((e**2).mean())*100:7.2f} {ch:7.2f} {sat:6.1f} {k1:8.1f}")
