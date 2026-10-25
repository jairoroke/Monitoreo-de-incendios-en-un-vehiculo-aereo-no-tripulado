#!/usr/bin/env python3
"""Compara FuegoNet con YOLO26n usando el MISMO código de métricas y las mismas imágenes.

    ~/.venvs/yolo/bin/python -m cuadrirotor_sim.red_fuego.evaluar --datos data.yaml \
        --fuegonet mejor.pt --yolo incendio_yolo26n.pt yolo26n_desde_cero.pt --salida eval.json

Mide, en la parte de validación: P, R, mAP50, mAP50-95 por clase, parámetros,
GFLOPs a 640 x 640 y la mediana de la latencia por imagen (lote 1) en GPU y CPU.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from .datos import a_pixeles, leer_etiquetas, listar
from .inferir import Detector
from .metricas import Acumulador
from .modelo import FuegoNet, contar_parametros


def gflops(modelo, disp):
    x = torch.zeros(1, 3, 640, 640, device=disp)
    with FlopCounterMode(display=False) as f:
        modelo(x)
    return f.get_total_flops() / 1e9          # una multiplicación-suma cuenta como 2 FLOPs


def latencia(fn, img, n=100):
    """Mediana en ms de n llamadas (la mediana ignora las pausas del sistema)."""
    for _ in range(15):
        fn(img)
    tiempos = []
    for _ in range(n):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(img)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tiempos.append(time.perf_counter() - t0)
    return float(np.median(tiempos)) * 1000


def evaluar(nombre, predecir, imgs, nc):
    acc = Acumulador(nc)
    for p in imgs:
        img = cv2.imread(p)
        h, w = img.shape[:2]
        e = leer_etiquetas(p)
        gt = np.concatenate((e[:, :1], a_pixeles(e, w, h)), 1)
        acc.agregar(predecir(img), gt)
    m = acc.calcular()
    print(f"{nombre:28s} P {m['P']:.3f} R {m['R']:.3f} mAP50 {m['mAP50']:.3f} mAP50-95 {m['mAP50-95']:.3f}")
    return m


def principal(a):
    imgs, nombres = listar(a.datos, 'val')
    nc = len(nombres)
    muestra = cv2.imread(imgs[0])
    res = {}
    modelos = []
    if a.fuegonet:
        for disp in ('0', 'cpu'):
            d = Detector(a.fuegonet, disp)
            modelos.append(('FuegoNet-' + torch.load(a.fuegonet, map_location='cpu')['escala'], disp,
                            lambda img, d=d: d(img, 0.001), d.modelo, d.disp))
    if a.yolo:
        from ultralytics import YOLO
        for ruta in a.yolo:
            for disp in ('0', 'cpu'):
                y = YOLO(ruta)
                f = lambda img, y=y, disp=disp: (lambda r: np.concatenate(
                    (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None],
                     r.boxes.cls.cpu().numpy()[:, None]), 1))(
                    y.predict(img, conf=0.001, imgsz=640, device=disp, verbose=False, max_det=300)[0])
                modelos.append((os.path.basename(ruta), disp, f, y.model, torch.device(
                    'cuda:0' if disp == '0' else 'cpu')))
    for nombre, disp, fn, red, dev in modelos:
        r = res.setdefault(nombre, {})
        if disp == '0':
            m = evaluar(nombre, fn, imgs, nc)
            r.update({k: (v.tolist() if hasattr(v, 'tolist') else float(v)) for k, v in m.items()})
            # Parámetros de la red entrenada (antes de fusionar BatchNorm y con las dos cabezas)
            r['parametros_M'] = (contar_parametros(FuegoNet(red.nc, red.escala)) if isinstance(red, FuegoNet)
                                 else sum(p.numel() for p in red.parameters())) / 1e6
            try:
                r['GFLOPs'] = gflops(red.to(dev).float().eval(), dev)
            except Exception as ex:          # algunas capas de ultralytics no se dejan contar
                print('  GFLOPs no disponibles:', ex)
            r['ms_gpu'] = latencia(fn, muestra)
        else:
            torch.set_num_threads(a.hilos)
            r['ms_cpu'] = latencia(fn, muestra)
        print(f"  {nombre} {disp}: {r.get('ms_gpu' if disp == '0' else 'ms_cpu'):.1f} ms")
    with open(a.salida, 'w') as f:
        json.dump(res, f, indent=2)
    print('Guardado', a.salida)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--datos', required=True)
    p.add_argument('--fuegonet', default='')
    p.add_argument('--yolo', nargs='*', default=[])
    p.add_argument('--salida', default='evaluacion.json')
    p.add_argument('--hilos', type=int, default=4)
    principal(p.parse_args())
