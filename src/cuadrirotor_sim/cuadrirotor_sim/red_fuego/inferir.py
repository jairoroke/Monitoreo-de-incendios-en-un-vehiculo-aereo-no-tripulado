#!/usr/bin/env python3
"""Usar FuegoNet entrenada: cargar los pesos y detectar en una imagen.

    ~/.venvs/yolo/bin/python -m cuadrirotor_sim.red_fuego.inferir mejor.pt foto.jpg [salida.jpg]

La clase Detector la usa también el nodo de ROS 2 (detector_yolo con un modelo
*.fuegonet.pt). Devuelve cajas en píxeles de la imagen ORIGINAL: deshace el
letterbox (escala r y relleno dx, dy).

Para inferir más rápido (como hace ultralytics):
  * cada BatchNorm se pliega en su convolución (fusionar), y
  * opcional (rect=True): la imagen se rellena solo hasta un múltiplo de 32. Es
    más rápido, pero FuegoNet pierde 0.013 de mAP50-95 (se entrenó con imágenes
    cuadradas); por omisión se usa el cuadrado de 640 x 640.
"""
import sys
import time

import cv2
import numpy as np
import torch

from .bloques import Conv
from .datos import letterbox
from .modelo import FuegoNet


@torch.no_grad()
def fusionar(modelo):
    """Pliega BatchNorm en la convolución: y = g (W x - mu) / sigma + b = W' x + b'."""
    for m in modelo.modules():
        if isinstance(m, Conv) and isinstance(m.bn, torch.nn.BatchNorm2d):
            bn, conv = m.bn, m.conv
            k = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            nueva = torch.nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                                    conv.padding, conv.dilation, conv.groups, bias=True).to(conv.weight.device)
            nueva.weight.copy_(conv.weight * k.view(-1, 1, 1, 1))
            nueva.bias.copy_(bn.bias - bn.running_mean * k)
            m.conv, m.bn = nueva, torch.nn.Identity()
    return modelo


class Detector:

    def __init__(self, pesos, dispositivo='0', rect=False):
        self.rect = rect
        self.disp = torch.device(f'cuda:{dispositivo}' if dispositivo.isdigit() and torch.cuda.is_available()
                                 else 'cpu')
        p = torch.load(pesos, map_location=self.disp)
        self.nombres = {int(k): v for k, v in p['nombres'].items()}
        self.imgsz = p.get('imgsz', 640)
        self.modelo = FuegoNet(p['nc'], p['escala']).to(self.disp).eval()
        self.modelo.load_state_dict(p['modelo'])
        fusionar(self.modelo)

    @torch.no_grad()
    def __call__(self, img_bgr, conf=0.25):
        """Lista de (x1, y1, x2, y2, confianza, clase) para una imagen BGR."""
        x, r, (dx, dy) = letterbox(img_bgr, self.imgsz, rect=self.rect)
        x = torch.from_numpy(np.ascontiguousarray(x[..., ::-1].transpose(2, 0, 1))).to(self.disp)
        x = x[None].float() / 255
        y = self.modelo(x)[0].float().cpu().numpy()
        y = y[y[:, 4] >= conf]                      # sin NMS: solo un umbral
        y[:, [0, 2]] = (y[:, [0, 2]] - dx) / r
        y[:, [1, 3]] = (y[:, [1, 3]] - dy) / r
        h, w = img_bgr.shape[:2]
        y[:, [0, 2]] = y[:, [0, 2]].clip(0, w)
        y[:, [1, 3]] = y[:, [1, 3]].clip(0, h)
        return y


class _Cajas:
    def __init__(self, d):
        t = torch.from_numpy(d)
        self.xyxy, self.conf, self.cls = t[:, :4], t[:, 4], t[:, 5]


class _Resultado:
    def __init__(self, d):
        self.boxes = _Cajas(d)


class ComoYolo:
    """Envoltura con la interfaz de ultralytics que usa detector_yolo:
    .names y .predict(img_bgr, conf=..., device=..., verbose=...)[0].boxes."""

    def __init__(self, pesos, dispositivo='0'):
        self.det = Detector(pesos, str(dispositivo))
        self.names = self.det.nombres

    def predict(self, img, conf=0.25, device=None, verbose=False):
        return [_Resultado(self.det(np.ascontiguousarray(img), conf))]


def dibujar(img, dets, nombres):
    colores = {0: (0, 60, 255), 1: (255, 200, 0)}
    for x1, y1, x2, y2, c, k in dets:
        col = colores.get(int(k), (255, 255, 255))
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(img, f'{nombres[int(k)]} {c:.2f}', (int(x1), max(14, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return img


if __name__ == '__main__':
    pesos, ruta = sys.argv[1], sys.argv[2]
    salida = sys.argv[3] if len(sys.argv) > 3 else 'deteccion.jpg'
    det = Detector(pesos)
    img = cv2.imread(ruta)
    det(img)                                        # calentar
    t0 = time.perf_counter()
    dets = det(img)
    print(f'{len(dets)} detecciones en {(time.perf_counter() - t0) * 1000:.1f} ms')
    for d in dets:
        print(f'  {det.nombres[int(d[5])]:6s} {d[4]:.2f}  caja ' + ' '.join(f'{v:.1f}' for v in d[:4]))
    cv2.imwrite(salida, dibujar(img, dets, det.nombres))
