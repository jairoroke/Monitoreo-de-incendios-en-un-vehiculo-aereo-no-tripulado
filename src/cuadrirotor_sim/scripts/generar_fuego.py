#!/usr/bin/env python3
"""Genera models/foco_incendio: un incendio con partículas de llama y humo.

Gazebo dibuja cada partícula con una textura (sprite) y le cambia el color a
lo largo de su vida según una imagen de rango de colores. Aquí se generan:

  materials/textures/llama.png          sprite de llama: lengua suave, alfa en los bordes
  materials/textures/llama_colores.png  blanco amarillento -> naranja -> rojo -> transparente
  materials/textures/humo.png           sprite de humo: nube difusa
  materials/textures/humo_colores.png   gris oscuro -> gris claro -> transparente

y el model.sdf con: varios emisores de llama repartidos en el área del foco,
una columna de humo, suelo quemado, troncos con brasas y una luz naranja.
El mundo necesita el sistema gz-sim-particle-emitter-system.

Uso:  python3 scripts/generar_fuego.py
"""
import math
import os
import random

import numpy as np
from PIL import Image

RADIO = 1.6          # radio del área que arde [m]
N_LLAMAS = 7         # emisores de llama
SEMILLA = 3


def sprite_llama(n=128):
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    # Lengua: ancha abajo, afilada arriba (y = 0 es arriba en la imagen).
    ancho = 0.12 + 0.30 * y**1.2
    d = np.abs(x - 0.5) / ancho
    alfa = np.clip(1 - d**2, 0, 1) * np.clip(y * 1.6, 0, 1) * np.clip((1 - y) * 4, 0, 1)
    alfa = alfa**1.5
    rgb = np.ones((n, n, 3))
    return np.dstack([rgb, alfa])


def sprite_humo(n=128, semilla=1):
    rnd = np.random.default_rng(semilla)
    y, x = np.mgrid[0:n, 0:n] / (n - 1) - 0.5
    r = np.hypot(x, y)
    alfa = np.clip(1 - (r / 0.5)**2, 0, 1)**1.8
    # Textura de nube: ruido suavizado a varias escalas.
    ruido = np.zeros((n, n))
    for esc in (4, 8, 16):
        g = rnd.random((esc + 1, esc + 1))
        img = Image.fromarray((g * 255).astype(np.uint8)).resize((n, n), Image.BICUBIC)
        ruido += np.asarray(img) / 255 / esc
    ruido = (ruido - ruido.min()) / (ruido.max() - ruido.min())
    alfa *= 0.55 + 0.45 * ruido
    rgb = np.ones((n, n, 3))
    return np.dstack([rgb, alfa])


def rango(colores, n=256):
    """Imagen 1 x n que interpola los colores RGBA dados (de inicio a fin de vida)."""
    t = np.linspace(0, 1, n)
    puntos = np.linspace(0, 1, len(colores))
    c = np.array(colores, dtype=float)
    fila = np.stack([np.interp(t, puntos, c[:, k]) for k in range(4)], axis=1)
    return np.repeat(fila[None, :, :], 4, axis=0)


def guardar(arr, ruta):
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), 'RGBA').save(ruta)


def emisor_llama(i, x, y):
    return f"""
      <particle_emitter name="llama_{i}" type="box">
        <pose>{x:.2f} {y:.2f} 0.05 0 -1.5707963 0</pose>
        <emitting>true</emitting>
        <size>0.25 0.5 0.5</size>
        <particle_size>0.7 0.7 0.7</particle_size>
        <lifetime>1.0</lifetime>
        <min_velocity>1.5</min_velocity>
        <max_velocity>2.8</max_velocity>
        <scale_rate>0</scale_rate>
        <rate>45</rate>
        <material>
          <diffuse>1 1 1</diffuse>
          <emissive>1 0.6 0.2</emissive>
          <pbr><metal><albedo_map>model://foco_incendio/materials/textures/llama.png</albedo_map></metal></pbr>
        </material>
        <color_range_image>model://foco_incendio/materials/textures/llama_colores.png</color_range_image>
      </particle_emitter>"""


def tronco(i, x, y, yaw):
    return f"""
      <visual name="tronco_{i}">
        <pose>{x:.2f} {y:.2f} 0.07 0 1.5707963 {yaw:.2f}</pose>
        <geometry><cylinder><radius>0.07</radius><length>1.0</length></cylinder></geometry>
        <material><ambient>0.04 0.03 0.03 1</ambient><diffuse>0.05 0.04 0.035 1</diffuse>
          <emissive>0.12 0.02 0 1</emissive></material>
      </visual>"""


def main():
    raiz = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'models', 'foco_incendio')
    tex = os.path.join(raiz, 'materials', 'textures')
    os.makedirs(tex, exist_ok=True)
    guardar(sprite_llama(), os.path.join(tex, 'llama.png'))
    guardar(sprite_humo(), os.path.join(tex, 'humo.png'))
    guardar(rango([(1, 0.95, 0.75, 0.9), (1, 0.7, 0.15, 0.95), (0.95, 0.3, 0.03, 0.8),
                   (0.5, 0.08, 0.02, 0.4), (0.15, 0.05, 0.03, 0.0)]),
            os.path.join(tex, 'llama_colores.png'))
    guardar(rango([(0.12, 0.11, 0.1, 0.0), (0.16, 0.15, 0.14, 0.75), (0.32, 0.31, 0.3, 0.55),
                   (0.55, 0.55, 0.55, 0.25), (0.7, 0.7, 0.7, 0.0)]),
            os.path.join(tex, 'humo_colores.png'))

    rnd = random.Random(SEMILLA)
    llamas, troncos, brasas = [], [], []
    # Manchas de brasa: discos pequeños de rojo oscuro a naranja, irregulares.
    for i in range(14):
        r = RADIO * 0.85 * math.sqrt(rnd.random())
        a = rnd.uniform(0, 2 * math.pi)
        k = rnd.uniform(0.25, 0.7)
        brasas.append(f"""
      <visual name="brasa_{i}">
        <pose>{r * math.cos(a):.2f} {r * math.sin(a):.2f} {0.015 + 0.001 * i:.3f} 0 0 0</pose>
        <geometry><cylinder><radius>{rnd.uniform(0.2, 0.45):.2f}</radius><length>0.005</length></cylinder></geometry>
        <material><ambient>0.1 0.03 0.01 1</ambient><diffuse>0.08 0.03 0.01 1</diffuse>
          <emissive>{0.12 + 0.3 * k:.2f} {0.015 + 0.05 * k:.3f} 0 1</emissive></material>
      </visual>""")
    for i in range(N_LLAMAS):
        r = RADIO * 0.75 * math.sqrt(rnd.random()) if i else 0.0
        a = rnd.uniform(0, 2 * math.pi)
        llamas.append(emisor_llama(i, r * math.cos(a), r * math.sin(a)))
    for i in range(6):
        a = i * math.pi / 3 + rnd.uniform(-0.3, 0.3)
        r = rnd.uniform(0.2, RADIO * 0.8)
        troncos.append(tronco(i, r * math.cos(a), r * math.sin(a), a + rnd.uniform(-0.6, 0.6)))

    sdf = f"""<?xml version="1.0"?>
<!-- Generado por scripts/generar_fuego.py; no editar a mano.
     Foco de incendio de {2 * RADIO:g} m de diámetro: {N_LLAMAS} emisores de llama, columna de
     humo, suelo quemado, troncos con brasas y luz. Requiere en el mundo el sistema
     gz-sim-particle-emitter-system. Solo visual: no colisiona. -->
<sdf version="1.9">
  <model name="foco_incendio">
    <static>true</static>
    <link name="link">
      <visual name="suelo_quemado">
        <pose>0 0 0.01 0 0 0</pose>
        <geometry><cylinder><radius>{RADIO * 1.5:.2f}</radius><length>0.01</length></cylinder></geometry>
        <material><ambient>0.06 0.05 0.04 1</ambient><diffuse>0.08 0.07 0.06 1</diffuse></material>
      </visual>
{''.join(brasas)}{''.join(troncos)}
{''.join(llamas)}

      <particle_emitter name="humo" type="box">
        <pose>0 0 1.4 0 -1.5707963 0</pose>
        <emitting>true</emitting>
        <size>0.5 {RADIO * 1.2:.2f} {RADIO * 1.2:.2f}</size>
        <particle_size>1.2 1.2 1.2</particle_size>
        <lifetime>9</lifetime>
        <min_velocity>0.7</min_velocity>
        <max_velocity>1.3</max_velocity>
        <scale_rate>0.9</scale_rate>
        <rate>9</rate>
        <material>
          <diffuse>1 1 1</diffuse>
          <pbr><metal><albedo_map>model://foco_incendio/materials/textures/humo.png</albedo_map></metal></pbr>
        </material>
        <color_range_image>model://foco_incendio/materials/textures/humo_colores.png</color_range_image>
      </particle_emitter>

      <light name="resplandor" type="point">
        <pose>0 0 0.8 0 0 0</pose>
        <diffuse>1 0.5 0.15 1</diffuse>
        <specular>0.3 0.15 0.05 1</specular>
        <attenuation><range>14</range><constant>0.3</constant><linear>0.12</linear><quadratic>0.02</quadratic></attenuation>
        <cast_shadows>false</cast_shadows>
        <intensity>3</intensity>
      </light>
    </link>
  </model>
</sdf>
"""
    with open(os.path.join(raiz, 'model.sdf'), 'w') as f:
        f.write(sdf)
    print(f'{raiz}: {N_LLAMAS} emisores de llama, humo, texturas en materials/textures')
    columna_humo(os.path.join(os.path.dirname(raiz), 'columna_humo'))


def columna_humo(raiz):
    """Penacho alto para el mundo 'bosque_grande': se ve sobre las copas desde lejos.

    Va en el mismo punto que foco_incendio y usa sus texturas. Sube ~40 m y se
    ensancha hasta ~15 m; sus partículas grandes y lentas tapan menos el foco
    que si se alargara la columna del propio foco_incendio.
    """
    os.makedirs(raiz, exist_ok=True)
    with open(os.path.join(raiz, 'model.sdf'), 'w') as f:
        f.write("""<?xml version="1.0"?>
<!-- Generado por scripts/generar_fuego.py; no editar a mano. Penacho de humo de ~40 m
     sobre foco_incendio (usa sus texturas). Solo visual. -->
<sdf version="1.9">
  <model name="columna_humo">
    <static>true</static>
    <link name="link">
      <particle_emitter name="penacho" type="box">
        <pose>0 0 5 0 -1.5707963 0</pose>
        <emitting>true</emitting>
        <size>1 3 3</size>
        <particle_size>3 3 3</particle_size>
        <lifetime>26</lifetime>
        <min_velocity>1.2</min_velocity>
        <max_velocity>1.8</max_velocity>
        <scale_rate>0.45</scale_rate>
        <rate>5</rate>
        <material>
          <diffuse>1 1 1</diffuse>
          <pbr><metal><albedo_map>model://foco_incendio/materials/textures/humo.png</albedo_map></metal></pbr>
        </material>
        <color_range_image>model://foco_incendio/materials/textures/humo_colores.png</color_range_image>
      </particle_emitter>
    </link>
  </model>
</sdf>
""")
    with open(os.path.join(raiz, 'model.config'), 'w') as f:
        f.write('<?xml version="1.0"?>\n<model><name>columna_humo</name>'
                '<sdf version="1.9">model.sdf</sdf></model>\n')
    print(f'{raiz}: penacho de humo')


if __name__ == '__main__':
    main()
