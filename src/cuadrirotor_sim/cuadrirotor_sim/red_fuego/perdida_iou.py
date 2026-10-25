"""IoU y CIoU entre cajas xyxy (mismo número de cajas en a y b)."""
import math

import torch


def iou(a, b, eps=1e-7):
    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[..., 2:] - a[..., :2]).prod(-1)
    area_b = (b[..., 2:] - b[..., :2]).prod(-1)
    return inter / (area_a + area_b - inter + eps)


def ciou(a, b, eps=1e-7):
    """Complete IoU: IoU - (distancia entre centros)^2 / diagonal^2 - alpha*v.

    El segundo término jala los centros aunque las cajas no se toquen (el IoU
    solo daría cero y ningún gradiente); el tercero iguala la proporción ancho/alto.
    """
    (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = a.unbind(-1), b.unbind(-1)
    w1, h1 = ax2 - ax1, (ay2 - ay1).clamp(min=eps)
    w2, h2 = bx2 - bx1, (by2 - by1).clamp(min=eps)
    i = iou(a, b, eps)
    cw = torch.maximum(ax2, bx2) - torch.minimum(ax1, bx1)
    ch = torch.maximum(ay2, by2) - torch.minimum(ay1, by1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((bx1 + bx2 - ax1 - ax2) ** 2 + (by1 + by2 - ay1 - ay2) ** 2) / 4
    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
    with torch.no_grad():
        alfa = v / (v - i + (1 + eps))
    return i - (rho2 / c2 + v * alfa)
