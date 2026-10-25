"""Función de pérdida de FuegoNet (la de YOLO26, sin DFL).

Para cada cabeza (o2m y o2o):
    L = 7.5 * L_caja + 0.5 * L_clase + 1.5 * L_dist
  L_clase  entropía cruzada binaria entre los logits y el puntaje suave de TAL,
           en TODAS las celdas (las de fondo deben dar cero).
  L_caja   1 - CIoU en las celdas de primer plano, pesada por su puntaje.
  L_dist   L1 entre las distancias (l, t, r, b) predichas y las reales,
           normalizadas por el tamaño de la imagen (sustituye a la DFL).

Las dos cabezas se combinan con un peso que cambia durante el entrenamiento
(ProgLoss de YOLO26):
    L_total = w * L_o2m + (1 - w) * L_o2o,   w: 0.8 -> 0.1 lineal en las épocas.
Al principio manda la o2m (muchas celdas positivas, aprende rápido); al final
manda la o2o, que es la que se usa para inferir.
"""
import torch
import torch.nn.functional as F

from .asignador import AsignadorTAL
from .modelo import anclas, dist_a_caja
from .perdida_iou import ciou


class PerdidaCabeza:

    def __init__(self, nc, pasos, topk, topk2=None, g_caja=7.5, g_clase=0.5, g_dist=1.5):
        self.nc, self.pasos = nc, pasos
        self.asignador = AsignadorTAL(nc, topk, topk2, lado_min=float(pasos[1]))
        self.g = (g_caja, g_clase, g_dist)

    def __call__(self, pred, etiquetas, imgsz):
        dist, logits = pred['dist'].float(), pred['logits'].float()
        B = dist.shape[0]
        puntos, paso = anclas(pred['mapas'], self.pasos)
        cajas_p = dist_a_caja(dist, puntos)                         # en celdas
        clases_gt, cajas_gt, valido = etiquetas
        cajas_obj, puntaje, fp = self.asignador(
            logits.detach().sigmoid(), (cajas_p.detach() * paso), puntos * paso,
            clases_gt, cajas_gt, valido)
        suma = puntaje.sum().clamp(min=1)
        l_clase = F.binary_cross_entropy_with_logits(logits, puntaje, reduction='none').sum() / suma
        l_caja = l_dist = dist.sum() * 0.0
        if fp.any():
            peso = puntaje[fp].sum(-1)
            obj = cajas_obj / paso                                  # a celdas
            l_caja = ((1.0 - ciou(cajas_p[fp], obj[fp])) * peso).sum() / suma
            # Distancias reales en celdas -> píxeles -> fracción de la imagen.
            p = puntos.expand(B, -1, -1)
            dist_obj = torch.cat((p - obj[..., :2], obj[..., 2:] - p), -1)
            esc = paso / imgsz
            l_dist = (F.l1_loss(dist[fp] * esc.expand(B, -1, -1)[fp],
                                dist_obj[fp] * esc.expand(B, -1, -1)[fp], reduction='none')
                      .mean(-1) * peso).sum() / suma
        perdidas = torch.stack((l_caja * self.g[0], l_clase * self.g[1], l_dist * self.g[2]))
        return perdidas.sum() * B, perdidas.detach()


class PerdidaE2E:
    """o2m + o2o con el peso progresivo de YOLO26."""

    def __init__(self, nc, pasos, epocas, w0=0.8, wf=0.1):
        self.o2m = PerdidaCabeza(nc, pasos, topk=10)
        self.o2o = PerdidaCabeza(nc, pasos, topk=7, topk2=1)
        self.epocas, self.w0, self.wf = epocas, w0, wf
        self.w = w0

    def fijar_epoca(self, e):
        self.w = max(1 - e / max(self.epocas - 1, 1), 0) * (self.w0 - self.wf) + self.wf

    def __call__(self, pred, lote):
        imgsz = lote['img'].shape[-1]
        etiquetas = rellenar(lote, imgsz)
        lm, _ = self.o2m(pred['o2m'], etiquetas, imgsz)
        lo, partes = self.o2o(pred['o2o'], etiquetas, imgsz)
        return self.w * lm + (1 - self.w) * lo, partes


def rellenar(lote, imgsz):
    """Lista de etiquetas (idx, clase, cx, cy, w, h normalizados) -> tensores (B, M, ·) en píxeles."""
    t = lote['etiquetas']
    B = lote['img'].shape[0]
    disp = lote['img'].device
    if t.shape[0] == 0:
        return (torch.zeros(B, 0, 1, device=disp), torch.zeros(B, 0, 4, device=disp),
                torch.zeros(B, 0, 1, device=disp))
    i = t[:, 0].long()
    M = int(torch.bincount(i, minlength=B).max())
    clases = torch.zeros(B, M, 1, device=disp)
    cajas = torch.zeros(B, M, 4, device=disp)
    valido = torch.zeros(B, M, 1, device=disp)
    orden = torch.cat([torch.arange(int(n), device=disp) for n in torch.bincount(i, minlength=B)])
    i, t = i.to(disp), t.to(disp)
    xywh = t[:, 2:6] * imgsz
    clases[i, orden, 0] = t[:, 1]
    cajas[i, orden] = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), -1)
    valido[i, orden, 0] = 1
    return clases, cajas, valido
