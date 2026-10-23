#!/usr/bin/env python3
"""Genera un dataset de detección de incendios (formato YOLO) desde Gazebo.

Abre el mundo 'bosque' sin dron, con una cámara de la misma óptica que la del
dron (640x480, campo horizontal 1.3 rad), y repite: mueve el foco de incendio y
la cámara a posiciones aleatorias, toma una imagen y la etiqueta proyectando el
volumen de las llamas ('fuego') y de la columna de humo ('humo') con el modelo
de cámara estenopeica. Los puntos del volumen que tapan las copas de los
árboles se descartan, así que cada caja cubre solo la parte visible.

Requiere Gazebo corriendo con el mundo que genera este mismo script:
    python3 scripts/capturar_dataset.py preparar /tmp/datos/bosque.sdf /tmp/cuadrirotor_sim_$UID/modelos
    GZ_SIM_RESOURCE_PATH=/tmp/datos/modelos_captura:.../models gz sim -s -r --headless-rendering /tmp/datos/bosque.sdf
    python3 scripts/capturar_dataset.py capturar ~/datasets/incendio 2500

Con el mundo 'bosque_grande' (relieve, rocas, animales, cielo y el penacho
alto) se agrega 'bosque_grande' al final de las dos órdenes:
    python3 scripts/capturar_dataset.py preparar /tmp/datos/bg.sdf /tmp/cuadrirotor_sim_$UID/modelos bosque_grande
    python3 scripts/capturar_dataset.py capturar ~/datasets/incendio_bosque_grande 3000 bosque_grande
Ahí las etiquetas descartan también lo que tapa el terreno, el foco se pone en
cualquier punto del valle y una de cada cinco tomas apunta a una roca o a un
animal (ejemplos de lo que no es fuego).
(Los modelos de /tmp/cuadrirotor_sim_$UID/modelos los deja el launch.)
"""
import math
import os
import random
import re
import sys
import threading
import time

import numpy as np
from gz.msgs10.image_pb2 import Image as GzImage
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean
from gz.transport13 import Node
from PIL import Image

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from cuadrirotor_sim import bosque_grande  # noqa: E402
ANCHO, ALTO, HFOV = 640, 480, 1.3
FX = (ANCHO / 2) / math.tan(HFOV / 2)
CLASES = ['fuego', 'humo']
FOCO_REAL = (8.0, 5.0)

# Copas de los árboles como cilindros: radio, altura inferior y superior [m].
COPA = {'roble': (4.0, 1.5, 6.5), 'pino': (1.4, 0.8, 5.1)}

CAMARA_SDF = f"""
    <model name="camara_datos">
      <pose>0 0 10 0 0.5 0</pose>
      <link name="link">
        <gravity>false</gravity>
        <inertial><mass>1</mass></inertial>
        <sensor name="camara" type="camera">
          <always_on>1</always_on>
          <update_rate>30</update_rate>
          <topic>/datos/camara</topic>
          <camera>
            <horizontal_fov>{HFOV}</horizontal_fov>
            <image><width>{ANCHO}</width><height>{ALTO}</height><format>R8G8B8</format></image>
            <clip><near>0.1</near><far>1500</far></clip>
          </camera>
        </sensor>
      </link>
    </model>
"""


def arboles(mundo='bosque'):
    """Árboles como (especie, x, y, z de la base, escala)."""
    if mundo == 'bosque_grande':
        return [(e, x, y, z, s) for e, x, y, z, _, s in
                bosque_grande.datos(os.path.join(RAIZ, 'models'))[0]['arboles']]
    s = open(os.path.join(RAIZ, 'worlds', 'bosque.sdf')).read()
    return [(e, float(x), float(y), 0.0, 1.0) for e, x, y in re.findall(
        r'model://(roble|pino)</uri>\s*<name>[^<]+</name>\s*<pose>([-\d.]+) ([-\d.]+)', s)]


def movil(sdf, nombre, nuevo):
    """Copia dinámica y sin gravedad de un modelo estático (set_pose no mueve estáticos)."""
    sdf = sdf.replace('<static>true</static>', '').replace(
        '<link name="link">', '<link name="link">\n      <gravity>false</gravity>\n'
        '      <inertial><mass>1</mass></inertial>', 1)
    return sdf.replace(f'<model name="{nombre}">', f'<model name="{nuevo}">')


def preparar(salida, modelos_fuego, mundo='bosque'):
    """Mundo de captura: el bosque con la cámara y un foco que se pueden mover.

    Gazebo no mueve modelos estáticos con set_pose, así que la cámara y las copias
    del foco ('foco_movil' y, en bosque_grande, 'humo_movil', en el directorio de
    salida) son dinámicos sin gravedad.
    """
    if mundo == 'bosque_grande':
        s = bosque_grande.sdf(os.path.join(RAIZ, 'models'))
        s = s.replace('<world name="bosque_grande">', '<world name="datos">')
        s = s.replace('<uri>model://columna_humo</uri>', '<uri>model://humo_movil</uri>')
    else:
        s = open(os.path.join(RAIZ, 'worlds', 'bosque.sdf')).read()
        s = s.replace('<world name="bosque">', '<world name="datos">')
    s = s.replace('<uri>model://foco_incendio</uri>', '<uri>model://foco_movil</uri>')
    s = s.replace('  </world>', CAMARA_SDF + '  </world>')
    with open(salida, 'w') as f:
        f.write(s)
    copias = {'foco_movil': 'foco_incendio'}
    if mundo == 'bosque_grande':
        copias['humo_movil'] = 'columna_humo'
    for nuevo, nombre in copias.items():
        sdf = open(os.path.join(modelos_fuego, nombre, 'model.sdf')).read()
        d = os.path.join(os.path.dirname(os.path.abspath(salida)), 'modelos_captura', nuevo)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'model.sdf'), 'w') as f:
            f.write(movil(sdf, nombre, nuevo))
        with open(os.path.join(d, 'model.config'), 'w') as f:
            f.write(f'<?xml version="1.0"?>\n<model><name>{nuevo}</name>'
                    '<sdf version="1.9">model.sdf</sdf></model>\n')
    print(f'Mundo de captura: {salida}; agrega {os.path.dirname(d)} a GZ_SIM_RESOURCE_PATH')


def puntos_volumen(fx, fy, fz=0.0, penacho=False):
    """Muestras 3D del volumen de las llamas y de la columna de humo.

    Con penacho (bosque_grande) la columna sigue hasta ~32 m y se ensancha.
    """
    ang = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    fuego = np.array([(fx + r * math.cos(a), fy + r * math.sin(a), fz + z)
                      for z in (0.05, 0.7, 1.4, 2.1) for r in (0.6, 1.6) for a in ang]
                     + [(fx, fy, fz + z) for z in (0.3, 1.0, 1.8)])
    alturas = np.linspace(2.2, 32.0, 14) if penacho else np.linspace(2.2, 8.0, 7)
    ancho = 0.10 if penacho else 0.32
    humo = np.array([(fx + (1.2 + ancho * (z - 2.2)) * math.cos(a),
                      fy + (1.2 + ancho * (z - 2.2)) * math.sin(a), fz + z)
                     for z in alturas for a in ang])
    return {'fuego': fuego, 'humo': humo}


def ocultos(c, P, troncos, terreno=None):
    """True para cada punto de P cuyo rayo desde la cámara c cruza una copa o el suelo."""
    tapado = np.zeros(len(P), bool)
    d = P - c
    for e, ax, ay, az, k in troncos:
        r, z0, z1 = (v * k for v in COPA[e])
        z0, z1 = z0 + az, z1 + az
        if np.min(np.hypot(P[:, 0] - ax, P[:, 1] - ay)) > 400 and math.hypot(c[0] - ax, c[1] - ay) > 400:
            continue
        # Intersección rayo-cilindro en el plano xy, luego se revisa la altura.
        ox, oy = c[0] - ax, c[1] - ay
        a = d[:, 0]**2 + d[:, 1]**2
        b = 2 * (ox * d[:, 0] + oy * d[:, 1])
        cc = ox**2 + oy**2 - r**2
        disc = b**2 - 4 * a * cc
        ok = (disc > 0) & (a > 1e-9)
        sq = np.sqrt(np.where(ok, disc, 0))
        for t in ((-b - sq) / (2 * np.where(a > 0, a, 1)), (-b + sq) / (2 * np.where(a > 0, a, 1))):
            z = c[2] + t * d[:, 2]
            tapado |= ok & (t > 0.02) & (t < 0.98) & (z > z0) & (z < z1)
    if terreno is not None and not terreno.plano:
        t = np.linspace(0.02, 0.97, 40)
        Q = c[None, None, :] + t[None, :, None] * d[:, None, :]
        tapado |= (Q[..., 2] < terreno.altura(Q[..., 0], Q[..., 1])).any(axis=1)
    return tapado


def proyectar(c, R, P):
    q = (P - c) @ R          # coordenadas en el marco de la cámara (x adelante)
    delante = q[:, 0] > 0.3
    u = ANCHO / 2 - FX * q[:, 1] / np.where(delante, q[:, 0], 1)
    v = ALTO / 2 - FX * q[:, 2] / np.where(delante, q[:, 0], 1)
    return u, v, delante


def rot(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = (math.cos(yaw), math.sin(yaw), math.cos(pitch),
                              math.sin(pitch), math.cos(roll), math.sin(roll))
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def etiquetas(c, R, foco, troncos, terreno=None):
    filas = []
    fz = terreno.altura(*foco) if terreno is not None else 0.0
    volumenes = puntos_volumen(*foco, fz, penacho=terreno is not None)
    for k, (nombre, P) in enumerate(volumenes.items()):
        u, v, delante = proyectar(c, R, P)
        visibles = delante & ~ocultos(c, P, troncos, terreno)
        if visibles.mean() < (0.4 if nombre == 'fuego' else 0.25):
            continue
        dentro = visibles & (u >= 0) & (u < ANCHO) & (v >= 0) & (v < ALTO)
        # El penacho alto suele salirse del cuadro por arriba; basta con que se vea una parte.
        if dentro.sum() < (0.1 if nombre == 'humo' and terreno is not None else 0.35) * visibles.sum():
            continue
        x0, x1 = np.clip([u[visibles].min(), u[visibles].max()], 0, ANCHO - 1)
        y0, y1 = np.clip([v[visibles].min(), v[visibles].max()], 0, ALTO - 1)
        if x1 - x0 < 5 or y1 - y0 < (5 if nombre == 'fuego' else 8):
            continue
        filas.append(f'{k} {(x0 + x1) / 2 / ANCHO:.6f} {(y0 + y1) / 2 / ALTO:.6f} '
                     f'{(x1 - x0) / ANCHO:.6f} {(y1 - y0) / ALTO:.6f}')
    return filas


class Capturador:

    def __init__(self):
        self.nodo = Node()
        self.cv = threading.Condition()
        self.n = 0
        self.img = None
        self.nodo.subscribe(GzImage, '/datos/camara', self._img)

    def _img(self, msg):
        with self.cv:
            self.img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy()
            self.n += 1
            self.cv.notify_all()

    def mover(self, nombre, x, y, z, yaw=0.0, pitch=0.0, roll=0.0):
        p = Pose()
        p.name = nombre
        p.position.x, p.position.y, p.position.z = x, y, z
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        p.orientation.w = cr * cp * cy + sr * sp * sy
        p.orientation.x = sr * cp * cy - cr * sp * sy
        p.orientation.y = cr * sp * cy + sr * cp * sy
        p.orientation.z = cr * cp * sy - sr * sp * cy
        for _ in range(10):   # el servicio tarda en estar disponible al arrancar
            ok, rep = self.nodo.request('/world/datos/set_pose', p, Pose, Boolean, 2000)
            if ok and rep.data:
                return True
            time.sleep(0.5)
        return False

    def foto(self, saltar=3, espera=3.0):
        with self.cv:
            n0 = self.n
            self.cv.wait_for(lambda: self.n >= n0 + saltar, timeout=espera)
            return self.img


def posicion_foco(rnd, troncos):
    if rnd.random() < 0.5:
        return FOCO_REAL
    while True:   # terreno abierto alrededor del bosque o claros en el borde
        x, y = rnd.uniform(-60, 75), rnd.uniform(-55, 70)
        libre = min(math.hypot(x - ax, y - ay) - COPA[e][0] for e, ax, ay, _, _ in troncos)
        if libre > 3.0:
            return (x, y)


def capturar(salida, total, semilla=11):
    rnd = random.Random(semilla)
    troncos = arboles()
    for p in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        os.makedirs(os.path.join(salida, p), exist_ok=True)
    cap = Capturador()
    if cap.foto(1, 15.0) is None:
        sys.exit('No llegan imágenes de /datos/camara: ¿está corriendo Gazebo con el mundo de captura?')
    conteo = {'fuego': 0, 'humo': 0, 'vacias': 0}
    foco = None
    for i in range(total):
        if i % 60 == 0:
            foco = posicion_foco(rnd, troncos)
            if not cap.mover('foco_incendio', foco[0], foco[1], 0.0):
                sys.exit('No se pudo mover el foco')
            time.sleep(10)   # que el humo (9 s de vida) llene la columna en el lugar nuevo
        # Distancia y altura como las de un dron buscando: de 4 a 70 m, y bajo
        # solo cerca del foco (si no, la cámara quedaría entre las copas).
        d = math.exp(rnd.uniform(math.log(4), math.log(70)))
        h = rnd.uniform(2.5, 12) if d < 9 else rnd.uniform(7.5, 25)
        az = rnd.uniform(0, 2 * math.pi)
        c = np.array([foco[0] + d * math.cos(az), foco[1] + d * math.sin(az), h])
        # Mirando hacia el foco con desvío (a veces queda fuera de cuadro) y la
        # inclinación de la cámara del dron (30°) más la del propio dron.
        yaw = math.atan2(foco[1] - c[1], foco[0] - c[0]) + rnd.gauss(0, 0.45)
        pitch = 0.5236 + rnd.uniform(-0.15, 0.15)
        roll = rnd.uniform(-0.1, 0.1)
        if not cap.mover('camara_datos', *c, yaw, pitch, roll):
            continue
        img = cap.foto()
        if img is None:
            continue
        guardar(salida, i, img, etiquetas(c, rot(yaw, pitch, roll), foco, troncos), conteo)
        if i % 100 == 0:
            print(f'{i}/{total} imágenes; {conteo}', flush=True)
    with open(os.path.join(salida, 'data.yaml'), 'w') as f:
        f.write(f'path: {os.path.abspath(salida)}\ntrain: images/train\nval: images/val\n'
                f'names:\n  0: fuego\n  1: humo\n')
    print(f'Listo: {total} imágenes en {salida}; {conteo}')


def guardar(salida, i, img, filas, conteo):
    parte = 'val' if i % 10 == 0 else 'train'
    nombre = f'{i:05d}'
    Image.fromarray(img).save(os.path.join(salida, 'images', parte, nombre + '.jpg'), quality=92)
    with open(os.path.join(salida, 'labels', parte, nombre + '.txt'), 'w') as f:
        f.write('\n'.join(filas) + ('\n' if filas else ''))
    for fila in filas:
        conteo[CLASES[int(fila[0])]] += 1
    conteo['vacias'] += not filas


def capturar_bosque_grande(salida, total, semilla=23):
    """Como capturar(), con relieve, penacho alto y tomas de rocas y animales."""
    rnd = random.Random(semilla)
    obj, ter = bosque_grande.datos(os.path.join(RAIZ, 'models'))
    troncos = arboles('bosque_grande')
    xmin, xmax, ymin, ymax = obj['zona']
    blancos = [(r[1], r[2]) for r in obj['rocas'] if xmin - 40 < r[1] < xmax + 40 and ymin - 40 < r[2] < ymax + 40]
    blancos += [(f[1], f[2]) for f in obj['fauna']]
    for p in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        os.makedirs(os.path.join(salida, p), exist_ok=True)
    cap = Capturador()
    if cap.foto(1, 15.0) is None:
        sys.exit('No llegan imágenes de /datos/camara: ¿está corriendo Gazebo con el mundo de captura?')
    conteo = {'fuego': 0, 'humo': 0, 'vacias': 0}
    foco = None
    for i in range(total):
        if i % 60 == 0:
            while True:   # en cualquier punto del valle, sin caer encima de un tronco
                foco = (rnd.uniform(xmin, xmax), rnd.uniform(ymin, ymax))
                if all(math.hypot(foco[0] - ax, foco[1] - ay) > 0.6 * COPA[e][0] * k
                       for e, ax, ay, _, k in troncos):
                    break
            fz = ter.altura(*foco)
            if not (cap.mover('foco_incendio', foco[0], foco[1], fz)
                    and cap.mover('columna_humo', foco[0], foco[1], fz)):
                sys.exit('No se pudo mover el foco')
            time.sleep(30)   # que se apague el penacho viejo (26 s de vida) y suba el nuevo
        if rnd.random() < 0.2:
            # Toma de una roca o un animal, como la vería el dron al pasar.
            bx, by = rnd.choice(blancos)
            d, az = rnd.uniform(10, 70), rnd.uniform(0, 2 * math.pi)
            cx, cy = bx + d * math.cos(az), by + d * math.sin(az)
            yaw = math.atan2(by - cy, bx - cx) + rnd.gauss(0, 0.2)
            h = rnd.uniform(12, 40)
        else:
            d = math.exp(rnd.uniform(math.log(4), math.log(220)))
            az = rnd.uniform(0, 2 * math.pi)
            cx, cy = foco[0] + d * math.cos(az), foco[1] + d * math.sin(az)
            yaw = math.atan2(foco[1] - cy, foco[0] - cx) + rnd.gauss(0, 0.45)
            h = rnd.uniform(2.5, 12) if d < 9 else rnd.uniform(12, 45)
        c = np.array([cx, cy, ter.altura(cx, cy) + h])
        pitch = 0.5236 + rnd.uniform(-0.15, 0.15)
        roll = rnd.uniform(-0.1, 0.1)
        if not cap.mover('camara_datos', *c, yaw, pitch, roll):
            continue
        img = cap.foto()
        if img is None:
            continue
        guardar(salida, i, img, etiquetas(c, rot(yaw, pitch, roll), foco, troncos, ter), conteo)
        if i % 100 == 0:
            print(f'{i}/{total} imágenes; {conteo}', flush=True)
    with open(os.path.join(salida, 'data.yaml'), 'w') as f:
        f.write(f'path: {os.path.abspath(salida)}\ntrain: images/train\nval: images/val\n'
                f'names:\n  0: fuego\n  1: humo\n')
    print(f'Listo: {total} imágenes en {salida}; {conteo}')


if __name__ == '__main__':
    mundo = sys.argv[-1] if sys.argv[-1] in ('bosque', 'bosque_grande') else 'bosque'
    if sys.argv[1] == 'preparar':
        preparar(sys.argv[2], sys.argv[3], mundo)
    elif mundo == 'bosque_grande':
        capturar_bosque_grande(sys.argv[2], int(sys.argv[3]))
    else:
        capturar(sys.argv[2], int(sys.argv[3]))
