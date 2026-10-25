"""Asignación de etiquetas alineada a la tarea (TAL) con el ajuste STAL de YOLO26.

En cada imagen hay A celdas (anclas) y M objetos etiquetados. Para entrenar hay
que decidir qué celdas «responden» por cada objeto. TAL usa la propia
predicción de la red:

    t = s^alpha * IoU^beta

donde s es la probabilidad que la red da a la clase correcta y IoU es el
traslape de su caja con la etiqueta. Por cada objeto se toman las topk celdas
con mayor t entre las que caen dentro de la caja. Si una celda queda con dos
objetos, se queda con el de mayor IoU.

  * Uno a muchos (o2m): topk = 10, varias celdas por objeto -> señal abundante.
  * Uno a uno  (o2o):   topk = 7 y luego topk2 = 1, una sola celda por objeto
    -> la red aprende a no duplicar cajas y no hace falta NMS.

STAL (Small-Target-Aware Label assignment): una caja con un lado menor que el
paso intermedio (16 px) se agranda a 16 px SOLO para buscar candidatas. Sin
esto, una brasa de 6 px puede no contener ningún centro de celda y nunca
recibir gradiente.
"""
import torch

from .perdida_iou import ciou


class AsignadorTAL:

    def __init__(self, nc, topk=10, topk2=None, alpha=0.5, beta=6.0, lado_min=16.0, eps=1e-9):
        self.nc, self.topk, self.topk2 = nc, topk, topk2 or topk
        self.alpha, self.beta, self.lado_min, self.eps = alpha, beta, lado_min, eps

    @torch.no_grad()
    def __call__(self, prob, cajas_p, puntos, clases_gt, cajas_gt, valido):
        """
        prob      (B, A, C)  probabilidades predichas (sigmoide)
        cajas_p   (B, A, 4)  cajas predichas xyxy en píxeles
        puntos    (A, 2)     centros de las celdas en píxeles
        clases_gt (B, M, 1)  clase de cada etiqueta
        cajas_gt  (B, M, 4)  cajas etiquetadas xyxy en píxeles
        valido    (B, M, 1)  1 si la etiqueta existe (el lote se rellena con ceros)

        Devuelve cajas_obj (B, A, 4), puntaje_obj (B, A, C) y primer_plano (B, A).
        """
        B, A, _ = prob.shape
        M = cajas_gt.shape[1]
        if M == 0:
            return torch.zeros_like(cajas_p), torch.zeros_like(prob), torch.zeros(B, A, dtype=torch.bool,
                                                                               device=prob.device)
        # 1) Candidatas: centros dentro de la caja (agrandada si es diminuta: STAL).
        dentro = self._dentro(puntos, cajas_gt) & valido.bool()              # (B, M, A)
        # 2) Métrica de alineación solo en las candidatas.
        b, m, a = dentro.nonzero(as_tuple=True)
        iou = torch.zeros(B, M, A, device=prob.device, dtype=cajas_p.dtype)
        iou[b, m, a] = ciou(cajas_gt[b, m], cajas_p[b, a]).clamp_(0).to(iou.dtype)
        s = torch.zeros_like(iou)
        s[b, m, a] = prob[b, a, clases_gt[b, m, 0].long()].to(s.dtype)
        t = s.pow(self.alpha) * iou.pow(self.beta)
        # 3) Las topk mejores celdas de cada objeto.
        idx = t.topk(self.topk, dim=-1).indices                               # (B, M, k)
        pos = torch.zeros_like(t, dtype=torch.int8).scatter_(-1, idx, 1)
        pos = pos * dentro * valido.to(torch.int8)
        # 4) Celda con varios objetos: gana el de mayor IoU.
        varios = (pos.sum(1, keepdim=True) > 1).expand(-1, M, -1)
        mejor = torch.zeros_like(pos).scatter_(1, iou.argmax(1, keepdim=True), 1)
        pos = torch.where(varios, mejor, pos)
        # 5) Uno a uno: de lo que quedó, solo la mejor celda de cada objeto.
        if self.topk2 != self.topk:
            idx2 = (t * pos).topk(self.topk2, dim=-1).indices
            pos = pos * torch.zeros_like(pos).scatter_(-1, idx2, 1)
        primer_plano, obj = pos.max(1)                                       # (B, A)
        # 6) Objetivos de cada celda.
        base = torch.arange(B, device=prob.device)[:, None] * M
        cajas_obj = cajas_gt.view(-1, 4)[obj + base]
        clase_obj = clases_gt.long().view(-1)[obj + base].clamp_(0)
        puntaje = torch.zeros(B, A, self.nc, device=prob.device)
        puntaje.scatter_(2, clase_obj[..., None], 1.0)
        puntaje *= primer_plano[..., None]
        # 7) Puntaje suave: t normalizado para que la mejor celda de cada objeto
        #    valga su IoU. La red aprende que la confianza debe reflejar la calidad.
        t = t * pos
        iou = iou * pos
        norm = (t * iou.amax(-1, keepdim=True) / (t.amax(-1, keepdim=True) + self.eps)).amax(1)
        return cajas_obj, puntaje * norm[..., None], primer_plano.bool()

    def _dentro(self, puntos, cajas):
        c = (cajas[..., :2] + cajas[..., 2:]) / 2
        wh = (cajas[..., 2:] - cajas[..., :2]).clamp(min=self.lado_min)      # STAL
        lt, rb = (c - wh / 2)[:, :, None], (c + wh / 2)[:, :, None]
        d = torch.cat((puntos - lt, rb - puntos), -1)                        # (B, M, A, 4)
        return d.amin(-1) > self.eps
