#!/usr/bin/env python3
"""Entrena FuegoNet desde cero (sin pesos preentrenados) con el dataset del simulador.

Uso (con el Python de ~/.venvs/yolo):
    ~/.venvs/yolo/bin/python -m cuadrirotor_sim.red_fuego.entrenar \
        --datos ~/datasets/combinado_incendio_bosque_incendio_bosque_grande/data.yaml \
        --epocas 150 --salida ~/datasets/fuegonet/n_150

La receta sigue a ultralytics para YOLO26: lote nominal 64 (se acumulan
gradientes si el lote real es menor), calentamiento de 3 épocas, tasa de
aprendizaje lineal de lr0 a lr0*lrf, MuSGD, EMA de los pesos, mosaico apagado en
las últimas 10 épocas y el mejor modelo según 0.1 mAP50 + 0.9 mAP50-95.
"""
import argparse
import csv
import math
import os
import random
import time
from copy import deepcopy

import numpy as np
import torch

from .datos import DatasetYolo, juntar
from .metricas import Acumulador
from .modelo import FuegoNet, contar_parametros
from .optimizador import MuSGD, grupos_de_parametros
from .perdida import PerdidaE2E


class EMA:
    """Promedio móvil exponencial de los pesos: un modelo «suavizado» que generaliza mejor."""

    def __init__(self, modelo, decaimiento=0.9999, tau=2000):
        self.m = deepcopy(modelo).eval()
        self.n, self.d, self.tau = 0, decaimiento, tau
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def actualizar(self, modelo):
        self.n += 1
        d = self.d * (1 - math.exp(-self.n / self.tau))
        msd = modelo.state_dict()
        for k, v in self.m.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


def cargador(ds, lote, mezclar, trabajadores):
    return torch.utils.data.DataLoader(ds, lote, shuffle=mezclar, num_workers=trabajadores,
                                       collate_fn=juntar, pin_memory=True, drop_last=mezclar,
                                       persistent_workers=trabajadores > 0)


@torch.no_grad()
def validar(modelo, carga, nc, disp, conf=0.001):
    modelo.eval()
    acc = Acumulador(nc)
    for lote in carga:
        img = lote['img'].to(disp, non_blocking=True).float() / 255
        with torch.autocast('cuda', torch.bfloat16, enabled=disp.type == 'cuda'):
            salida = modelo(img).float().cpu().numpy()
        s = img.shape[-1]
        e = lote['etiquetas'].numpy()
        for i, pred in enumerate(salida):
            pred = pred[pred[:, 4] >= conf]
            g = e[e[:, 0] == i]
            xyxy = np.concatenate((g[:, 2:4] - g[:, 4:6] / 2, g[:, 2:4] + g[:, 4:6] / 2), 1) * s
            acc.agregar(pred, np.concatenate((g[:, 1:2], xyxy), 1))
    return acc.calcular()


def principal(a):
    random.seed(a.semilla)
    np.random.seed(a.semilla)
    torch.manual_seed(a.semilla)
    torch.backends.cudnn.benchmark = True
    disp = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.salida, exist_ok=True)

    ds_ent = DatasetYolo(a.datos, 'train', a.imgsz, aumentar=True, mosaico=a.mosaico)
    ds_val = DatasetYolo(a.datos, 'val', a.imgsz)
    nombres = ds_ent.nombres
    nc = len(nombres)
    carga_ent = cargador(ds_ent, a.lote, True, a.trabajadores)
    carga_val = cargador(ds_val, a.lote * 2, False, a.trabajadores)

    modelo = FuegoNet(nc, a.escala).to(disp)
    if a.pesos:
        modelo.load_state_dict(torch.load(a.pesos, map_location=disp)['modelo'])
    print(f'FuegoNet-{a.escala}: {contar_parametros(modelo) / 1e6:.2f} M parámetros, '
          f'{len(ds_ent)} imágenes de entrenamiento, {len(ds_val)} de validación')

    nominal = 64
    acumular = max(round(nominal / a.lote), 1)
    wd = a.wd * a.lote * acumular / nominal
    grupos = grupos_de_parametros(modelo, wd, muon=a.optimizador == 'musgd')
    if a.optimizador == 'adamw':
        opt = torch.optim.AdamW(grupos, lr=a.lr0, betas=(0.9, 0.999))
    elif a.optimizador == 'sgd':
        opt = torch.optim.SGD(grupos, lr=a.lr0, momentum=0.9, nesterov=True)
    else:
        opt = MuSGD(grupos, lr=a.lr0, momento=0.9)
    for g in opt.param_groups:
        g['lr_inicial'] = a.lr0
    ley = lambda e: (1 - e / a.epocas) * (1.0 - a.lrf) + a.lrf     # lineal lr0 -> lr0*lrf
    perdida = PerdidaE2E(nc, modelo.cabeza.pasos, a.epocas)
    ema = EMA(modelo)

    n_lotes = len(carga_ent)
    n_cal = max(round(3 * n_lotes), 100)            # iteraciones de calentamiento
    mejor, ultima_opt = -1.0, -1
    reg = open(os.path.join(a.salida, 'resultados.csv'), 'w', newline='')
    esc = csv.writer(reg)
    esc.writerow(['epoca', 'tiempo', 'caja', 'clase', 'dist', 'P', 'R', 'mAP50', 'mAP50-95',
                  'AP50-95_fuego', 'AP50-95_humo', 'lr', 'w_o2m'])
    t0 = time.time()
    for epoca in range(a.epocas):
        if epoca == a.epocas - a.sin_mosaico and a.mosaico > 0:
            print('Mosaico apagado para las últimas épocas')
            ds_ent.mosaico = 0.0
            carga_ent = cargador(ds_ent, a.lote, True, a.trabajadores)
        modelo.train()
        perdida.fijar_epoca(epoca)
        suma = torch.zeros(3, device=disp)
        opt.zero_grad(set_to_none=True)
        for k, lote in enumerate(carga_ent):
            it = k + n_lotes * epoca
            if it <= n_cal:   # calentamiento: lr de 0 (sesgos de 0.1) al valor de la ley
                x = it / n_cal
                for j, g in enumerate(opt.param_groups):
                    inicio = 0.1 if (j == 2 and a.optimizador != 'adamw') else 0.0
                    g['lr'] = np.interp(x, [0, 1], [inicio, g['lr_inicial'] * ley(epoca)])
                    if 'momentum' in g:
                        g['momentum'] = np.interp(x, [0, 1], [0.8, 0.9])
            lote['img'] = lote['img'].to(disp, non_blocking=True).float() / 255
            with torch.autocast('cuda', torch.bfloat16, enabled=disp.type == 'cuda'):
                pred = modelo(lote['img'])
            l, partes = perdida(pred, lote)
            (l / acumular).backward()
            suma += partes
            if (k + 1) % acumular == 0 or k + 1 == n_lotes:
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 10.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                ema.actualizar(modelo)
        for g in opt.param_groups:
            g['lr'] = g['lr_inicial'] * ley(epoca + 1)
        m = validar(ema.m, carga_val, nc, disp)
        suma = (suma / n_lotes).tolist()
        apto = 0.1 * m['mAP50'] + 0.9 * m['mAP50-95']
        fila = [epoca + 1, round(time.time() - t0, 1), *[round(x, 4) for x in suma],
                *[round(float(m[x]), 4) for x in ('P', 'R', 'mAP50', 'mAP50-95')],
                *[round(float(x), 4) for x in m['AP50-95_clase']],
                round(opt.param_groups[0]['lr'], 6), round(perdida.w, 3)]
        esc.writerow(fila)
        reg.flush()
        print(f"[{epoca + 1:3d}/{a.epocas}] pérdida caja {suma[0]:.3f} clase {suma[1]:.3f} "
              f"dist {suma[2]:.3f} | mAP50 {m['mAP50']:.3f} mAP50-95 {m['mAP50-95']:.3f} "
              f"| {time.time() - t0:.0f} s", flush=True)
        punto = {'modelo': ema.m.state_dict(), 'nc': nc, 'escala': a.escala, 'nombres': nombres,
                 'epoca': epoca + 1, 'metricas': {k: (v.tolist() if hasattr(v, 'tolist') else v)
                                                  for k, v in m.items()}, 'imgsz': a.imgsz}
        torch.save(punto, os.path.join(a.salida, 'ultimo.pt'))
        if apto > mejor:
            mejor, ultima_opt = apto, epoca
            torch.save(punto, os.path.join(a.salida, 'mejor.pt'))
        if epoca - ultima_opt >= a.paciencia:
            print(f'Sin mejora en {a.paciencia} épocas: se detiene')
            break
    reg.close()
    print(f'Listo en {(time.time() - t0) / 60:.1f} min. Mejor modelo: {os.path.join(a.salida, "mejor.pt")}')


def argumentos(args=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--datos', required=True, help='data.yaml del dataset')
    p.add_argument('--salida', required=True, help='carpeta para pesos y resultados.csv')
    p.add_argument('--escala', default='n', choices=['n', 's', 'm'])
    p.add_argument('--epocas', type=int, default=150)
    p.add_argument('--lote', type=int, default=32)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--optimizador', default='musgd', choices=['musgd', 'sgd', 'adamw'])
    p.add_argument('--lr0', type=float, default=0.01)
    p.add_argument('--lrf', type=float, default=0.01)
    p.add_argument('--wd', type=float, default=5e-4)
    p.add_argument('--mosaico', type=float, default=1.0)
    p.add_argument('--sin-mosaico', type=int, default=10, help='épocas finales sin mosaico')
    p.add_argument('--paciencia', type=int, default=50)
    p.add_argument('--trabajadores', type=int, default=8)
    p.add_argument('--pesos', default='', help='continuar desde un mejor.pt/ultimo.pt')
    p.add_argument('--semilla', type=int, default=0)
    return p.parse_args(args)


if __name__ == '__main__':
    principal(argumentos())
