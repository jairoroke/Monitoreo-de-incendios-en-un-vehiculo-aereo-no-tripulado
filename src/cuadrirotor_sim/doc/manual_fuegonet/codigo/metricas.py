"""Precisión media (mAP) como en COCO: mAP50 y mAP50-95.

Para un umbral de IoU t, una predicción es verdadero positivo (VP) si traslapa
con IoU >= t a una etiqueta de su misma clase que nadie más ha tomado (se
recorren de mayor a menor confianza). Con las predicciones ordenadas por
confianza se calculan precisión P = VP / (VP + FP) y exhaustividad
R = VP / (número de etiquetas); AP es el área bajo la curva P(R) con la
envolvente monótona, integrada en 101 puntos. mAP50-95 promedia AP en
t = 0.50, 0.55, ..., 0.95 y en las clases.
"""
import numpy as np

UMBRALES = np.linspace(0.5, 0.95, 10)


def iou_matriz(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    aa = (a[:, 2:] - a[:, :2]).prod(1)
    ab = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (aa[:, None] + ab[None] - inter + 1e-9)


class Acumulador:
    """Junta las predicciones de todas las imágenes y al final calcula mAP."""

    def __init__(self, nc):
        self.nc = nc
        self.vp, self.conf, self.clase, self.n_gt = [], [], [], np.zeros(nc, int)

    def agregar(self, pred, gt):
        """pred (N, 6) x1 y1 x2 y2 conf clase; gt (M, 5) clase x1 y1 x2 y2 (mismos píxeles)."""
        for c in gt[:, 0].astype(int):
            self.n_gt[c] += 1
        vp = np.zeros((len(pred), len(UMBRALES)), bool)
        if len(pred) and len(gt):
            orden = np.argsort(-pred[:, 4])
            pred = pred[orden]
            iou = iou_matriz(pred[:, :4], gt[:, 1:])
            iou[pred[:, 5][:, None].astype(int) != gt[:, 0][None].astype(int)] = 0
            for k, t in enumerate(UMBRALES):
                tomada = np.zeros(len(gt), bool)
                for i in range(len(pred)):
                    cand = np.where((iou[i] >= t) & ~tomada)[0]
                    if len(cand):
                        j = cand[np.argmax(iou[i, cand])]
                        tomada[j] = vp[i, k] = True
        elif len(pred):
            pred = pred[np.argsort(-pred[:, 4])]
        self.vp.append(vp)
        self.conf.append(pred[:, 4] if len(pred) else np.zeros(0))
        self.clase.append(pred[:, 5].astype(int) if len(pred) else np.zeros(0, int))

    def calcular(self):
        vp, conf, clase = np.concatenate(self.vp), np.concatenate(self.conf), np.concatenate(self.clase)
        ap = np.zeros((self.nc, len(UMBRALES)))
        p_r = np.zeros((self.nc, 2))
        for c in range(self.nc):
            m = clase == c
            if self.n_gt[c] == 0 or not m.any():
                continue
            o = np.argsort(-conf[m])
            v = vp[m][o]
            acum_vp = v.cumsum(0)
            acum_fp = (~v).cumsum(0)
            rec = acum_vp / self.n_gt[c]
            prec = acum_vp / (acum_vp + acum_fp)
            for k in range(len(UMBRALES)):
                ap[c, k] = area_pr(rec[:, k], prec[:, k])
            # P y R al umbral de confianza 0.25 (lo que usa el detector en vuelo)
            sel = conf[m][o] >= 0.25
            if sel.any():
                n = sel.sum() - 1
                p_r[c] = prec[n, 0], rec[n, 0]
        return {'mAP50': ap[:, 0].mean(), 'mAP50-95': ap.mean(), 'AP50-95_clase': ap.mean(1),
                'AP50_clase': ap[:, 0], 'P': p_r[:, 0].mean(), 'R': p_r[:, 1].mean()}


def area_pr(rec, prec):
    r = np.concatenate(([0.0], rec, [1.0]))
    p = np.concatenate(([1.0], prec, [0.0]))
    p = np.flip(np.maximum.accumulate(np.flip(p)))      # envolvente decreciente
    x = np.linspace(0, 1, 101)
    return np.trapz(np.interp(x, r, p), x)
