#!/usr/bin/env python3
"""Descarga y prepara los animales del mundo 'bosque_grande' (models/fauna_*).

Modelos de baja poligonización con licencia CC0 de OpenGameArt:
  venado  "Deer low poly rigged" de crownjoshua
  zorro   "Red Fox" del "Animals Pack" de Quaternius
  lobo    "Wolf" del "Animal Pack Vol. 2" de Quaternius

Los OBJ vienen con el eje Y hacia arriba y escalas arbitrarias; aquí se giran a
Z arriba con la cabeza hacia +x, se escalan a su tamaño real y se pone el origen
entre las patas, a ras del suelo. Cada modelo es dinámico, sin gravedad ni
colisión, para que el nodo 'fauna' lo pueda mover con set_pose.

Uso (con trimesh; en ~/.venvs/cad):
    ~/.venvs/cad/bin/python scripts/preparar_fauna.py /directorio/de/descargas
"""
import os
import sys
import urllib.request
import zipfile

import numpy as np
import trimesh

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OGA = 'https://opengameart.org/sites/default/files/'

# nombre: (zip, archivo OBJ dentro del zip, eje de la cabeza en el OBJ, altura real [m], página)
ANIMALES = {
    'venado': ('Deer%28Updated%29.zip', 'Deer/Other formats/Deer.obj', '+z', 1.55,
               'https://opengameart.org/content/deer-low-poly-rigged'),
    'zorro': ('Animals%20Pack%20by%20Quaternius.zip', 'Animals Pack by Quaternius/OBJ/Red Fox.obj',
              '-z', 0.55, 'https://opengameart.org/content/5-low-poly-animals'),
    'lobo': ('Animal%20Pack%20Vol.2%20by%20%40Quaternius.zip',
             'Animal Pack Vol.2 by @Quaternius/OBJ/Wolf.obj', '+x', 0.95,
             'https://opengameart.org/content/animated-animales-low-poly'),
}

# Y arriba -> Z arriba y cabeza hacia +x.
GIRO = {'+z': np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
        '-z': np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
        '+x': np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])}

# Colores por parte cuando el .mtl no los trae (el zorro y el lobo vienen en gris).
COLORES = {'zorro': [[0.86, 0.82, 0.74], [0.78, 0.36, 0.10]],
           'lobo': [[0.42, 0.40, 0.37]]}


def extraer(zip_, descargas):
    """Descarga el zip (una vez) y lo extrae en su propio directorio."""
    nombre = zip_.replace('%', '_')
    ruta, carpeta = os.path.join(descargas, nombre), os.path.join(descargas, nombre[:-4])
    if not os.path.exists(ruta):
        print(f'Descargando {OGA + zip_}')
        with urllib.request.urlopen(OGA + zip_, timeout=60) as r, open(ruta, 'wb') as f:
            f.write(r.read())
    if not os.path.isdir(carpeta):
        zipfile.ZipFile(ruta).extractall(carpeta)
    return carpeta


def preparar(nombre, descargas):
    zip_, obj, cabeza, altura, pagina = ANIMALES[nombre]
    escena = trimesh.load(os.path.join(extraer(zip_, descargas), obj), force='scene')
    M = np.eye(4)
    M[:3, :3] = GIRO[cabeza]
    escena.apply_transform(M)
    lo, hi = escena.bounds
    s = altura / (hi[2] - lo[2])
    T = np.eye(4)
    T[:3, :3] *= s
    T[:3, 3] = -s * np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    escena.apply_transform(T)

    destino = os.path.join(RAIZ, 'models', f'fauna_{nombre}')
    os.makedirs(os.path.join(destino, 'meshes'), exist_ok=True)
    visuales = []
    for i, malla in enumerate(escena.dump(concatenate=False)):
        kd = COLORES[nombre][i] if nombre in COLORES else malla.visual.material.main_color[:3] / 255
        archivo = f'{nombre}_{i}.stl'
        malla.export(os.path.join(destino, 'meshes', archivo))
        c = ' '.join(f'{x:.3f}' for x in kd)
        visuales.append(f"""
      <visual name="v{i}">
        <geometry><mesh><uri>meshes/{archivo}</uri></mesh></geometry>
        <material><ambient>{c} 1</ambient><diffuse>{c} 1</diffuse><specular>0.05 0.05 0.05 1</specular></material>
      </visual>""")
    largo = escena.bounds[1][0] - escena.bounds[0][0]
    with open(os.path.join(destino, 'model.sdf'), 'w') as f:
        f.write(f"""<?xml version="1.0"?>
<!-- Generado por scripts/preparar_fauna.py; no editar a mano. Solo visual: sin
     gravedad ni colisión, lo mueve el nodo 'fauna'. -->
<sdf version="1.9">
  <model name="fauna_{nombre}">
    <link name="link">
      <gravity>false</gravity>
      <inertial><mass>1</mass></inertial>{''.join(visuales)}
    </link>
  </model>
</sdf>
""")
    with open(os.path.join(destino, 'model.config'), 'w') as f:
        f.write(f'<?xml version="1.0"?>\n<model><name>fauna_{nombre}</name>'
                f'<sdf version="1.9">model.sdf</sdf></model>\n')
    with open(os.path.join(destino, 'ORIGEN.txt'), 'w') as f:
        f.write(f'Modelo de OpenGameArt ({pagina}), licencia CC0 (dominio público).\n'
                f'Archivo original: {obj} de {OGA + zip_}\n'
                f'Girado a Z arriba con la cabeza hacia +x y escalado a {altura} m de alto '
                f'por scripts/preparar_fauna.py.\n')
    print(f'{destino}: {len(visuales)} partes, alto {altura} m, largo {largo:.2f} m')


def main(descargas):
    os.makedirs(descargas, exist_ok=True)
    for nombre in ANIMALES:
        preparar(nombre, descargas)


if __name__ == '__main__':
    main(sys.argv[1])
