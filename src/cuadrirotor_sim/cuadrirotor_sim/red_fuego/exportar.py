#!/usr/bin/env python3
"""Exporta FuegoNet a ONNX (para la computadora a bordo: Jetson, Raspberry Pi, etc.).

    ~/.venvs/yolo/bin/python -m cuadrirotor_sim.red_fuego.exportar mejor.pt fuegonet_n.onnx [foto.jpg]

Como la cabeza no necesita NMS, el grafo exportado termina en un solo tensor
(1, 300, 6) = x1, y1, x2, y2, confianza, clase sobre la imagen 640 x 640 con
letterbox. Con onnxruntime instalado se comprueba que ONNX y PyTorch coinciden
sobre una foto (si se da) o una imagen al azar.
"""
import sys

import numpy as np
import torch

from .datos import letterbox
from .modelo import FuegoNet


def exportar(pesos, salida, foto=None):
    p = torch.load(pesos, map_location='cpu')
    m = FuegoNet(p['nc'], p['escala']).eval()
    m.load_state_dict(p['modelo'])
    x = torch.rand(1, 3, p.get('imgsz', 640), p.get('imgsz', 640))
    torch.onnx.export(m, (x,), salida, input_names=['imagen'], output_names=['detecciones'],
                      opset_version=17, dynamo=False)
    print('Exportado', salida)
    try:
        import onnxruntime as ort
    except ImportError:
        print('(onnxruntime no está instalado: no se comprueba)')
        return
    if foto:
        import cv2
        img = letterbox(cv2.imread(foto), x.shape[-1])[0][..., ::-1].transpose(2, 0, 1)
        x = torch.from_numpy(np.ascontiguousarray(img))[None].float() / 255
    s = ort.InferenceSession(salida, providers=['CPUExecutionProvider'])
    y_onnx = s.run(None, {'imagen': x.numpy()})[0]
    with torch.no_grad():
        y = m(x).numpy()
    # Las puntuaciones casi empatadas (cerca de cero) pueden salir en otro orden del
    # topk; se comparan solo las detecciones con confianza >= 0.25.
    k = int((y[0, :, 4] >= 0.25).sum())
    d = np.abs(y_onnx[0, :k, :5] - y[0, :k, :5]).max() if k else 0.0
    print(f'{k} detecciones con confianza >= 0.25; diferencia máxima ONNX vs PyTorch: {d:.2e}')


if __name__ == '__main__':
    exportar(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'fuegonet.onnx',
             sys.argv[3] if len(sys.argv) > 3 else None)
