#!/usr/bin/env python3
"""Genera el mundo 'bosque_grande': un valle boscoso rodeado de montañas.

  - Terreno de 1.2 x 1.2 km (malla de 4 m): valle ondulado de 420 x 420 m con
    tres cerros, y una cordillera alrededor que sube hasta ~250 m, con roca en las
    laderas empinadas y nieve en las cumbres. Textura propia de 4096 x 4096.
  - Unos 1700 pinos y robles (OpenRobotics, CC0) con claros de pastizal, de
    tamaño y orientación variables; rocas y peñascos generados aquí; venados,
    lobos y zorros (OpenGameArt, CC0; ver scripts/preparar_fauna.py).
  - Base de despegue al suroeste del valle, fuera del bosque, y el foco de
    incendio por omisión en una ladera al noreste.

Escribe models/terreno_bosque_grande/ (malla, textura, DEM altura.npz, rocas y
objetos.json) y worlds/bosque_grande.sdf. Semilla fija: siempre el mismo mundo.

Uso (necesita trimesh, Pillow y SciPy; están en ~/.venvs/cad):
    ~/.venvs/cad/bin/python scripts/generar_bosque_grande.py
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
import trimesh

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from cuadrirotor_sim import bosque_grande  # noqa: E402
from cuadrirotor_sim.terreno import Terreno  # noqa: E402

SEMILLA = 21
MEDIO = 600.0            # el terreno va de -MEDIO a MEDIO en x e y [m]
PASO = 4.0               # malla del terreno [m]
TEXTURA = 4096           # píxeles por lado
ZONA = (-210.0, 210.0, -210.0, 210.0)   # bosque que debe recorrer el dron
BASE = (-232.0, -232.0)
FOCO = (105.0, 135.0)
N_ARBOLES = 1700
N_ROCAS = 140
CERROS = [(60, 95, 42, 48), (-120, 40, 28, 38), (150, -110, 18, 30)]   # x, y, alto, sigma
DESTINO = os.path.join(RAIZ, 'models', 'terreno_bosque_grande')


def suave(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


class Ruido:
    """Ruido de valor suave en [-1, 1] y su suma fractal (fbm)."""

    def __init__(self, semilla):
        self.rng = np.random.default_rng(semilla)
        self.redes = {}

    def valor(self, x, y, escala):
        if escala not in self.redes:
            n = int(2 * MEDIO / escala) + 3
            self.redes[escala] = self.rng.uniform(-1, 1, (n, n))
        g = self.redes[escala]
        gx, gy = (x + MEDIO) / escala, (y + MEDIO) / escala
        ix, iy = np.floor(gx).astype(int), np.floor(gy).astype(int)
        u, v = suave(gx - ix), suave(gy - iy)
        return ((g[iy, ix] * (1 - u) + g[iy, ix + 1] * u) * (1 - v)
                + (g[iy + 1, ix] * (1 - u) + g[iy + 1, ix + 1] * u) * v)

    def fbm(self, x, y, escala, octavas=4):
        total, amp, norma = 0.0, 1.0, 0.0
        for k in range(octavas):
            total = total + amp * self.valor(x, y, escala / 2 ** k)
            norma += amp
            amp *= 0.5
        return total / norma


_BORDE = None


def distancia_al_valle(x, y):
    """Distancia (aprox.) al borde del valle: un cuadrado redondeado de 250 m de
    semilado con el contorno deformado por ruido para que no se vea artificial."""
    global _BORDE
    if _BORDE is None:
        _BORDE = Ruido(SEMILLA + 1)
    return (np.abs(x) ** 6 + np.abs(y) ** 6) ** (1 / 6) - 250.0 + 50.0 * _BORDE.fbm(x, y, 160.0, 3)


def alturas(rz):
    ejes = np.arange(-MEDIO, MEDIO + PASO / 2, PASO)
    x, y = np.meshgrid(ejes, ejes)          # h[iy, ix]
    h = 7.0 * rz.fbm(x, y, 160.0)
    for cx, cy, alto, sig in CERROS:
        forma = 1 + 0.25 * rz.fbm(x, y, 40.0, 3)
        h += alto * forma * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sig ** 2))
    d = distancia_al_valle(x, y)
    cresta = 1 - np.abs(rz.fbm(x, y, 280.0, 3))          # crestas afiladas
    h += suave(d / 260.0) * (70.0 + 130.0 * cresta ** 2) + 15.0 * suave(d / 70.0)
    h = gaussian_filter(h, 1.2, mode='nearest')    # suaviza los pliegues de las crestas
    rb = np.hypot(x - BASE[0], y - BASE[1])
    k = suave((rb - 16.0) / 34.0)
    h = h * k + (1 - k) * 0.0
    return ejes, h


def colores_textura(ter, rz, densidad):
    n = TEXTURA
    px = 2 * MEDIO / n
    c = -MEDIO + (np.arange(n) + 0.5) * px
    X, Y = np.meshgrid(c, c[::-1])          # fila 0 = y máximo (arriba de la imagen)
    Z = ter.altura(X, Y)
    gy, gx = np.gradient(Z, -px, px)
    incl = np.hypot(gx, gy)                 # tangente de la pendiente
    n1 = rz.fbm(X, Y, 60.0, 4)
    n2 = rz.fbm(X, Y, 9.0, 3)
    n3 = rz.valor(X, Y, 1.5)

    verde = np.array([0.29, 0.38, 0.17])
    verde2 = np.array([0.36, 0.43, 0.20])
    seco = np.array([0.50, 0.47, 0.29])
    monte = np.array([0.15, 0.22, 0.11])
    sotobosque = np.array([0.19, 0.23, 0.11])
    tierra = np.array([0.36, 0.30, 0.20])
    roca = np.array([0.46, 0.43, 0.39])
    roca2 = np.array([0.33, 0.31, 0.29])
    nieve = np.array([0.93, 0.94, 0.97])

    t = suave(0.5 + 0.9 * n1)[..., None]
    col = verde * (1 - t) + verde2 * t
    s = 0.6 * suave((n2 - 0.4) * 2.5)[..., None]
    col = col * (1 - s) + seco * s
    # Sotobosque oscuro bajo las copas y algo de tierra suelta.
    dens = map_coordinates(densidad, [(Y + MEDIO) / PASO, (X + MEDIO) / PASO], order=1)
    b = np.clip(dens, 0, 1)[..., None]
    col = col * (1 - b) + (sotobosque * (1 - 0.4 * suave(n2)[..., None]) + tierra * 0.4 * suave(n2)[..., None]) * b
    # Bosque denso (sin árboles 3D) en las faldas de las montañas.
    dv = distancia_al_valle(X, Y)
    m = (suave(dv / 40.0) * (1 - suave((dv - 120 - 60 * n1) / 40)) * suave(0.6 + 1.5 * n2))[..., None]
    col = col * (1 - m) + monte * (1 + 0.3 * n2[..., None]) * m
    # Roca en laderas empinadas y en lo alto; nieve en las cumbres.
    r = np.maximum(suave((incl - 0.8) / 0.35), suave((Z - 140 - 30 * n1) / 45))[..., None]
    tr = suave(0.5 + n2)[..., None]
    col = col * (1 - r) + (roca * (1 - tr) + roca2 * tr) * r
    w = (suave((Z - 185 - 20 * n1) / 20) * (1 - suave((incl - 1.1) / 0.4)))[..., None]
    col = col * (1 - w) + nieve * w
    col *= (1 + 0.07 * n3)[..., None]
    return (np.clip(col, 0, 1) * 255).astype(np.uint8)


def escribir_obj(ruta, ejes, h, mtl):
    n = len(ejes)
    X, Y = np.meshgrid(ejes, ejes)
    gy, gx = np.gradient(h, PASO)
    nrm = np.stack([-gx, -gy, np.ones_like(h)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    u = (X + MEDIO) / (2 * MEDIO)
    v = (Y + MEDIO) / (2 * MEDIO)
    idx = np.arange(n * n).reshape(n, n) + 1
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()      # (ix, iy), (ix+1, iy)
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()        # (ix, iy+1), (ix+1, iy+1)
    caras = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
    with open(ruta, 'w') as f:
        f.write('# Generado por scripts/generar_bosque_grande.py\n')
        f.write(f'mtllib {mtl}\n')
        np.savetxt(f, np.stack([X.ravel(), Y.ravel(), h.ravel()], 1), fmt='v %.2f %.2f %.3f')
        np.savetxt(f, np.stack([u.ravel(), v.ravel()], 1), fmt='vt %.5f %.5f')
        np.savetxt(f, nrm.reshape(-1, 3), fmt='vn %.4f %.4f %.4f')
        f.write('usemtl suelo\n')
        np.savetxt(f, np.repeat(caras, 3, axis=1), fmt='f %d/%d/%d %d/%d/%d %d/%d/%d')


def rocas_mallas(rng, n=6):
    rutas = []
    for k in range(n):
        m = trimesh.creation.icosphere(subdivisions=2)
        v = m.vertices
        dirs = rng.normal(size=(5, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        bulto = sum(rng.uniform(0.08, 0.2) * np.maximum(0, v @ dd) ** 3 for dd in dirs)
        r = 1 + bulto + rng.uniform(-0.06, 0.06, len(v))
        v = v * r[:, None] * np.array([1.0, rng.uniform(0.65, 0.95), rng.uniform(0.5, 0.8)])
        v[:, 2] = np.maximum(v[:, 2], -0.35)                 # base plana
        v[:, 2] -= v[:, 2].min()
        m.vertices = v / np.abs(v[:, :2]).max()              # ~1 m de semieje
        m.merge_vertices()
        ruta = os.path.join(DESTINO, 'meshes', f'roca_{k}.obj')
        m.export(ruta, include_normals=True)   # sin normales, la colisión de DART (ODE) se cae
        rutas.append(ruta)
    return len(rutas)


def distribuir_arboles(rng, rz, ter):
    arboles, celdas = [], {}

    def vecinos(x, y):
        i, j = int(x // 10), int(y // 10)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                yield from celdas.get((i + di, j + dj), [])

    intentos = 0
    while len(arboles) < N_ARBOLES and intentos < 400000:
        intentos += 1
        x, y = rng.uniform(ZONA[0] - 20, ZONA[1] + 20), rng.uniform(ZONA[2] - 20, ZONA[3] + 20)
        d = distancia_al_valle(x, y)
        if rng.random() > 1 - suave((d + 40) / 50):           # se ralea hacia las montañas
            continue
        claro = rz.fbm(x, y, 90.0, 3)
        if claro < -0.22 and rng.random() < 0.97:              # pastizales
            continue
        if math.hypot(x - BASE[0], y - BASE[1]) < 30:
            continue
        z = ter.altura(x, y)
        incl = math.hypot(ter.altura(x + 1, y) - ter.altura(x - 1, y),
                          ter.altura(x, y + 1) - ter.altura(x, y - 1)) / 2
        if incl > 0.75 or z > 95:
            continue
        pino = rng.random() < 0.45 + 0.4 * suave(z / 45)       # más pinos en lo alto
        especie = 'pino' if pino else 'roble'
        s = rng.uniform(0.9, 1.6) if pino else rng.uniform(0.8, 1.25)
        r = bosque_grande.ESPECIES[especie]['medidas'][0] * s
        if any(math.hypot(x - ax, y - ay) < 0.75 * (r + ar) for ax, ay, ar in vecinos(x, y)):
            continue
        arboles.append([especie, round(x, 2), round(y, 2), round(float(z), 2),
                        round(rng.uniform(0, 2 * math.pi), 3), round(s, 3)])
        celdas.setdefault((int(x // 10), int(y // 10)), []).append((x, y, r))
    return arboles


def distribuir_rocas(rng, ter, arboles, n_mallas):
    rocas = []
    while len(rocas) < N_ROCAS:
        x, y = rng.uniform(-330, 330), rng.uniform(-330, 330)
        d = distancia_al_valle(x, y)
        if d > 70 or math.hypot(x - BASE[0], y - BASE[1]) < 25:
            continue
        en_cerro = max(alto * math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sig ** 2))
                       for cx, cy, alto, sig in CERROS) / 40
        if d > -10:
            if rng.random() > 0.3:
                continue
            s = rng.uniform(3.0, 7.0)          # peñascos al pie de las montañas
        elif rng.random() < 0.35 + en_cerro:
            s = rng.uniform(1.2, 4.0)
        else:
            continue
        if any(math.hypot(x - a[1], y - a[2]) < s + 2 for a in arboles):
            continue
        ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        z = float(np.min(ter.altura(x + s * np.cos(ang), y + s * np.sin(ang)))) - 0.15 * s
        g = rng.uniform(0.17, 0.27)
        color = [round(g + rng.uniform(0.01, 0.04), 3), round(g, 3), round(g - rng.uniform(0.0, 0.03), 3)]
        rocas.append([int(rng.integers(n_mallas)), round(x, 2), round(y, 2), round(z, 2),
                      round(rng.uniform(-0.15, 0.15), 3), round(rng.uniform(-0.15, 0.15), 3),
                      round(rng.uniform(0, 2 * math.pi), 3), round(s, 2), color])
    return rocas


def distribuir_fauna(rng, rz, ter, arboles):
    """Manadas de venados, parejas de lobos y zorros solitarios en los claros."""
    fauna = []
    grupos = [('venado', 3)] * 4 + [('lobo', 2)] * 2 + [('zorro', 1)] * 5
    for tipo, n in grupos:
        while True:
            cx, cy = rng.uniform(ZONA[0] + 20, ZONA[1] - 20), rng.uniform(ZONA[2] + 20, ZONA[3] - 20)
            if rz.fbm(cx, cy, 90.0, 3) < -0.25 and math.hypot(cx - BASE[0], cy - BASE[1]) > 60 \
                    and all(math.hypot(cx - a[1], cy - a[2]) > 6 for a in arboles):
                break
        for _ in range(n):
            fauna.append([tipo, round(cx + rng.uniform(-4, 4), 2), round(cy + rng.uniform(-4, 4), 2),
                          round(rng.uniform(0, 2 * math.pi), 3)])
    return fauna


def main():
    rng = np.random.default_rng(SEMILLA)
    rz = Ruido(SEMILLA)
    os.makedirs(os.path.join(DESTINO, 'meshes'), exist_ok=True)
    os.makedirs(os.path.join(DESTINO, 'materials', 'textures'), exist_ok=True)

    ejes, h = alturas(rz)
    dem = os.path.join(DESTINO, 'altura.npz')
    np.savez_compressed(dem, x0=-MEDIO, y0=-MEDIO, paso=PASO, h=h.astype(np.float32))
    ter = Terreno(dem)
    print(f'Terreno: {len(ejes)}x{len(ejes)} vértices, alturas {h.min():.1f} a {h.max():.1f} m')

    arboles = distribuir_arboles(rng, rz, ter)
    n_mallas = rocas_mallas(rng)
    rocas = distribuir_rocas(rng, ter, arboles, n_mallas)
    fauna = distribuir_fauna(rng, rz, ter, arboles)

    # Densidad de copas sobre la malla del terreno, para oscurecer el sotobosque.
    densidad = np.zeros_like(h)
    for _, x, y, _, _, s in arboles:
        iy, ix = int(round((y + MEDIO) / PASO)), int(round((x + MEDIO) / PASO))
        densidad[iy, ix] += s
    densidad = np.clip(gaussian_filter(densidad, 1.3) * 6, 0, 1)
    Image.fromarray(colores_textura(ter, rz, densidad)).save(
        os.path.join(DESTINO, 'materials', 'textures', 'suelo.jpg'), quality=90)

    escribir_obj(os.path.join(DESTINO, 'meshes', 'terreno.obj'), ejes, h, 'terreno.mtl')
    with open(os.path.join(DESTINO, 'meshes', 'terreno.mtl'), 'w') as f:
        f.write('newmtl suelo\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nNs 1\nillum 1\n'
                'map_Kd ../materials/textures/suelo.jpg\n')

    with open(os.path.join(DESTINO, 'model.sdf'), 'w') as f:
        f.write("""<?xml version="1.0"?>
<!-- Generado por scripts/generar_bosque_grande.py; no editar a mano. -->
<sdf version="1.9">
  <model name="terreno_bosque_grande">
    <static>true</static>
    <link name="link">
      <collision name="col">
        <geometry><mesh><uri>meshes/terreno.obj</uri></mesh></geometry>
      </collision>
      <visual name="vis">
        <geometry><mesh><uri>meshes/terreno.obj</uri></mesh></geometry>
        <material>
          <diffuse>1 1 1 1</diffuse><specular>0 0 0 1</specular>
          <pbr><metal><albedo_map>materials/textures/suelo.jpg</albedo_map>
            <roughness>1.0</roughness><metalness>0.0</metalness></metal></pbr>
        </material>
      </visual>
    </link>
  </model>
</sdf>
""")
    with open(os.path.join(DESTINO, 'model.config'), 'w') as f:
        f.write('<?xml version="1.0"?>\n<model><name>terreno_bosque_grande</name>'
                '<sdf version="1.9">model.sdf</sdf></model>\n')
    with open(os.path.join(DESTINO, 'objetos.json'), 'w') as f:
        json.dump({'zona': ZONA, 'base': BASE, 'foco': FOCO, 'arboles': arboles,
                   'rocas': rocas, 'fauna': fauna}, f, separators=(',', ':'))

    mundo = os.path.join(RAIZ, 'worlds', 'bosque_grande.sdf')
    bosque_grande.escribir(os.path.join(RAIZ, 'models'), mundo)
    n_p = sum(a[0] == 'pino' for a in arboles)
    print(f'{mundo}: {n_p} pinos, {len(arboles) - n_p} robles, {len(rocas)} rocas, '
          f'{len(fauna)} animales; foco en {FOCO}')


if __name__ == '__main__':
    main()
