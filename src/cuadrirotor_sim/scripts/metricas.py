#!/usr/bin/env python3
"""Resume corridas guardadas en CSV.  Uso: scripts/metricas.py corrida1.csv [corrida2.csv ...]

Columnas: error de posición RMS en la órbita sin viento (13-15 s) y con viento
(t > 25 s), error máximo con viento, índice de chattering (media de |Δω| de los
rotores por muestra) y ganancia k1 final del STA.
"""
import sys

import numpy as np
import pandas as pd

print(f"{'corrida':28s} {'rms sin viento':>14} {'rms viento':>11} {'max viento':>11} "
      f"{'chattering':>10} {'k1 final (x,y,z)':>20}")
for f in sys.argv[1:]:
    d = pd.read_csv(f)
    e = np.sqrt((d.x - d.xd)**2 + (d.y - d.yd)**2 + (d.z - d.zd)**2)
    a, b = (d.t > 13) & (d.t < 15), d.t > 25
    w = d[['w0', 'w1', 'w2', 'w3']].values
    ch = np.abs(np.diff(w, axis=0))[b.values[1:]].mean()
    k = f"({d.k1x.iloc[-1]:.2f},{d.k1y.iloc[-1]:.2f},{d.k1z.iloc[-1]:.2f})" if 'k1x' in d else '-'
    print(f"{f.split('/')[-1][:-4]:28s} {np.sqrt((e[a]**2).mean()):14.4f} {np.sqrt((e[b]**2).mean()):11.4f} "
          f"{e[b].max():11.4f} {ch:10.2f} {k:>20}")
