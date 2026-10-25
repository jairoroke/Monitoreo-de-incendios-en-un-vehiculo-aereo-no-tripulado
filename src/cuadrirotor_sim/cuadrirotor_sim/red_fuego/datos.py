"""Lectura del dataset en formato YOLO y aumentos de datos.

Formato: images/{train,val}/xxx.jpg y labels/{train,val}/xxx.txt, una línea por
objeto «clase cx cy w h» normalizados a [0, 1] (lo que escribe
scripts/capturar_dataset.py).

Aumentos de entrenamiento (los mismos que ultralytics por omisión):
  mosaico     cuatro imágenes en un lienzo 2s x 2s alrededor de un centro al azar;
  afín        escala 1 ± 0.5 y traslación ± 10 %;
  HSV         tono ± 0.015, saturación ± 0.5, brillo ± 0.4 (la luz del
              simulador es fija: esto evita que la red dependa del color exacto);
  espejo      horizontal con probabilidad 0.5.
El mosaico se apaga en las últimas épocas para que la red vea imágenes «normales».
"""
import glob
import os
import random

import cv2
import numpy as np
import torch
import yaml

EXT = ('.jpg', '.jpeg', '.png', '.bmp')


def listar(data_yaml, parte):
    """Rutas de imágenes de 'train' o 'val' (acepta una ruta o una lista)."""
    with open(data_yaml) as f:
        d = yaml.safe_load(f)
    base = d.get('path', os.path.dirname(os.path.abspath(data_yaml)))
    rutas = d[parte] if isinstance(d[parte], list) else [d[parte]]
    imgs = []
    for r in rutas:
        r = r if os.path.isabs(r) else os.path.join(base, r)
        imgs += sorted(p for p in glob.glob(os.path.join(r, '*')) if p.lower().endswith(EXT))
    return imgs, d['names']


def ruta_etiqueta(img):
    a, b = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep
    return os.path.splitext(b.join(img.rsplit(a, 1)))[0] + '.txt'


def leer_etiquetas(img):
    p = ruta_etiqueta(img)
    if not os.path.exists(p) or os.path.getsize(p) == 0:   # imagen sin fuego ni humo
        return np.zeros((0, 5), np.float32)
    return np.loadtxt(p, ndmin=2, dtype=np.float32).reshape(-1, 5)


def letterbox(img, s, color=114, rect=False):
    """Escala para que el lado mayor mida s y rellena hasta s x s (sin deformar).

    Con rect=True solo rellena hasta el siguiente múltiplo de 32 (el paso mayor de
    la red): una imagen de 640 x 480 queda igual y se procesan 25 % menos píxeles.
    """
    h, w = img.shape[:2]
    r = s / max(h, w)
    nw, nh = round(w * r), round(h * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if r != 1 else img
    W, H = ((-(-nw // 32) * 32), (-(-nh // 32) * 32)) if rect else (s, s)
    dx, dy = (W - nw) / 2, (H - nh) / 2
    img = cv2.copyMakeBorder(img, round(dy - 0.1), round(dy + 0.1), round(dx - 0.1), round(dx + 0.1),
                             cv2.BORDER_CONSTANT, value=(color,) * 3)
    return img, r, (dx, dy)


def a_pixeles(e, w, h, dx=0, dy=0):
    """cx cy w h normalizados -> x1 y1 x2 y2 en píxeles (con desplazamiento)."""
    c = e[:, 1:].copy()
    xy = np.stack((c[:, 0] - c[:, 2] / 2, c[:, 1] - c[:, 3] / 2,
                   c[:, 0] + c[:, 2] / 2, c[:, 1] + c[:, 3] / 2), 1)
    return xy * [w, h, w, h] + [dx, dy, dx, dy]


class DatasetYolo(torch.utils.data.Dataset):

    def __init__(self, data_yaml, parte, imgsz=640, aumentar=False, mosaico=1.0, escala=0.5,
                 traslacion=0.1, hsv=(0.015, 0.5, 0.4), espejo=0.5):
        self.imgs, self.nombres = listar(data_yaml, parte)
        if not self.imgs:
            raise FileNotFoundError(f'No hay imágenes de {parte} en {data_yaml}')
        self.etiquetas = [leer_etiquetas(p) for p in self.imgs]
        self.s, self.aumentar, self.mosaico = imgsz, aumentar, mosaico
        self.escala, self.traslacion, self.hsv, self.espejo = escala, traslacion, hsv, espejo

    def __len__(self):
        return len(self.imgs)

    def _cargar(self, i):
        """Imagen con lado mayor = s y sus cajas xyxy en píxeles de esa imagen."""
        img = cv2.imread(self.imgs[i])
        h, w = img.shape[:2]
        r = self.s / max(h, w)
        if r != 1:
            img = cv2.resize(img, (round(w * r), round(h * r)), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
        e = self.etiquetas[i]
        return img, e[:, 0].copy(), a_pixeles(e, w, h)

    def _mosaico(self, i):
        s = self.s
        xc, yc = (int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2))
        lienzo = np.full((2 * s, 2 * s, 3), 114, np.uint8)
        clases, cajas = [], []
        for k, j in enumerate([i] + random.choices(range(len(self)), k=3)):
            img, c, b = self._cargar(j)
            h, w = img.shape[:2]
            if k == 0:   # arriba a la izquierda del centro
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif k == 1:  # arriba a la derecha
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * s), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif k == 2:  # abajo a la izquierda
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(2 * s, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:         # abajo a la derecha
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * s), min(2 * s, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            lienzo[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            clases.append(c)
            cajas.append(b + [x1a - x1b, y1a - y1b, x1a - x1b, y1a - y1b])
        return lienzo, np.concatenate(clases), np.concatenate(cajas), -s // 2

    def _afin(self, img, clases, cajas, borde=0):
        """Escala y traslación al azar; la salida mide s x s."""
        s = self.s
        h, w = img.shape[:2]
        C = np.eye(3)
        C[0, 2], C[1, 2] = -w / 2, -h / 2                           # centro al origen
        R = np.eye(3)
        R[:2, :2] *= random.uniform(1 - self.escala, 1 + self.escala)
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - self.traslacion, 0.5 + self.traslacion) * s
        T[1, 2] = random.uniform(0.5 - self.traslacion, 0.5 + self.traslacion) * s
        M = (T @ R @ C)[:2]
        img = cv2.warpAffine(img, M, (s, s), borderValue=(114, 114, 114))
        if len(cajas):
            esq = np.ones((len(cajas) * 4, 3))
            esq[:, :2] = cajas[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(-1, 2)
            esq = (esq @ M.T).reshape(-1, 8)
            nuevas = np.stack((esq[:, 0::2].min(1), esq[:, 1::2].min(1),
                               esq[:, 0::2].max(1), esq[:, 1::2].max(1)), 1).clip(0, s)
            # Descartar cajas que quedaron casi fuera de la imagen.
            area0 = (cajas[:, 2:] - cajas[:, :2]).prod(1) * R[0, 0] ** 2
            wh = nuevas[:, 2:] - nuevas[:, :2]
            ok = (wh > 2).all(1) & (wh.prod(1) / (area0 + 1e-9) > 0.1)
            clases, cajas = clases[ok], nuevas[ok]
        return img, clases, cajas

    def _hsv(self, img):
        g = np.random.uniform(-1, 1, 3) * self.hsv + 1
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        x = np.arange(256, dtype=np.float32)
        lut_h = ((x * g[0]) % 180).astype(np.uint8)
        lut_s = np.clip(x * g[1], 0, 255).astype(np.uint8)
        lut_v = np.clip(x * g[2], 0, 255).astype(np.uint8)
        hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def __getitem__(self, i):
        if self.aumentar and random.random() < self.mosaico:
            img, clases, cajas, _ = self._mosaico(i)
            img, clases, cajas = self._afin(img, clases, cajas)
        else:
            img = cv2.imread(self.imgs[i])
            h, w = img.shape[:2]
            img, r, (dx, dy) = letterbox(img, self.s)
            e = self.etiquetas[i]
            clases, cajas = e[:, 0].copy(), a_pixeles(e, w * r, h * r, dx, dy)
            if self.aumentar:
                img, clases, cajas = self._afin(img, clases, cajas)
        if self.aumentar:
            img = self._hsv(img)
            if random.random() < self.espejo:
                img = np.ascontiguousarray(img[:, ::-1])
                cajas = cajas.copy()
                cajas[:, [0, 2]] = self.s - cajas[:, [2, 0]]
        # Salida: imagen RGB CHW uint8 y etiquetas (clase, cx, cy, w, h) normalizadas.
        cajas = np.asarray(cajas, np.float32).reshape(-1, 4)
        xywh = np.concatenate(((cajas[:, :2] + cajas[:, 2:]) / 2, cajas[:, 2:] - cajas[:, :2]), 1) / self.s
        e = np.concatenate((np.asarray(clases, np.float32).reshape(-1, 1), xywh), 1)
        img = torch.from_numpy(np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1)))
        return img, torch.from_numpy(e)


def juntar(lote):
    """Une las etiquetas del lote en una tabla (índice de imagen, clase, cx, cy, w, h)."""
    imgs, etqs = zip(*lote)
    filas = [torch.cat((torch.full((len(e), 1), float(i)), e), 1) for i, e in enumerate(etqs)]
    return {'img': torch.stack(imgs), 'etiquetas': torch.cat(filas)}
