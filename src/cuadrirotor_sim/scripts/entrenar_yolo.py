#!/usr/bin/env python3
"""Ajusta YOLO26 a las clases 'fuego' y 'humo' con el dataset del simulador.

Parte de los pesos preentrenados en COCO (yolo26n.pt, el modelo más rápido de la
familia, el recomendado para drones) y entrena con el dataset que genera
scripts/capturar_dataset.py. Copia el mejor modelo a yolo/incendio_yolo26n.pt
del paquete, que es el que usa detector_yolo.

Uso (con el Python de ~/.venvs/yolo):
    ~/.venvs/yolo/bin/python scripts/entrenar_yolo.py ~/datasets/incendio_bosque [épocas]

Con varios datasets se entrena con todos juntos (el modelo sirve en los dos
mundos, 'bosque' y 'bosque_grande'):
    ~/.venvs/yolo/bin/python scripts/entrenar_yolo.py ~/datasets/incendio_bosque \
        ~/datasets/incendio_bosque_grande 60
"""
import os
import shutil
import sys

from ultralytics import YOLO

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def combinar(datasets):
    """data.yaml con los train/val de todos los datasets; devuelve su directorio."""
    rutas = [os.path.abspath(d) for d in datasets]
    if len(rutas) == 1:
        return rutas[0]
    destino = os.path.join(os.path.dirname(rutas[0]),
                           'combinado_' + '_'.join(os.path.basename(r) for r in rutas))
    os.makedirs(destino, exist_ok=True)
    lista = lambda parte: ''.join(f'\n  - {os.path.join(r, "images", parte)}' for r in rutas)
    with open(os.path.join(destino, 'data.yaml'), 'w') as f:
        f.write(f'train:{lista("train")}\nval:{lista("val")}\nnames:\n  0: fuego\n  1: humo\n')
    return destino


def main(datasets, epocas=60):
    dataset = combinar(datasets)
    datos = os.path.join(dataset, 'data.yaml')
    modelo = YOLO('yolo26n.pt')
    modelo.train(data=datos, epochs=epocas, imgsz=640, batch=32, device=0, workers=8,
                 project=os.path.join(os.path.abspath(dataset), 'entrenamientos'),
                 name='yolo26n_incendio', exist_ok=True, patience=20,
                 # Aumentos: el simulador tiene luz fija; variar color y escala.
                 hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, scale=0.5, mosaic=1.0, fliplr=0.5)
    mejor = os.path.join(os.path.abspath(dataset), 'entrenamientos', 'yolo26n_incendio',
                         'weights', 'best.pt')
    metricas = YOLO(mejor).val(data=datos, imgsz=640, device=0, verbose=False,
                               project=os.path.join(os.path.abspath(dataset), 'entrenamientos'),
                               name='validacion', exist_ok=True)
    print(f'mAP50 = {metricas.box.map50:.3f}, mAP50-95 = {metricas.box.map:.3f}')
    for i, nombre in metricas.names.items():
        print(f'  {nombre}: mAP50-95 = {metricas.box.maps[i]:.3f}')
    destino = os.path.join(RAIZ, 'yolo', 'incendio_yolo26n.pt')
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    shutil.copy(mejor, destino)
    print(f'Modelo copiado a {destino}')


if __name__ == '__main__':
    args = sys.argv[1:]
    epocas = int(args.pop()) if args[-1].isdigit() else 60
    main(args, epocas)
