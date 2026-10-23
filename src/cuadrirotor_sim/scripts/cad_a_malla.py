#!/usr/bin/env python3
"""Convierte un ensamble STEP en mallas STL para Gazebo (unidades: metros).

Usa el teselador de OpenCASCADE (cascadio) y trimesh, que no vienen con ROS:
    python3 -m venv --without-pip ~/.venvs/cad
    curl -sSL https://bootstrap.pypa.io/get-pip.py | ~/.venvs/cad/bin/python
    ~/.venvs/cad/bin/pip install cascadio trimesh numpy fast_simplification

Uso:
    # 1) Ver las piezas: nombre, cuántas hay y su tamaño.
    ~/.venvs/cad/bin/python scripts/cad_a_malla.py listar dron.step

    # 2) Exportar grupos de piezas elegidas por nombre (expresión regular).
    #    'resto' toma las que no caen en otro grupo; --excluir las omite.
    #    --separar exporta cada pieza del grupo por aparte, centrada en su caja.
    ~/.venvs/cad/bin/python scripts/cad_a_malla.py exportar dron.step salida/ \\
        --grupo motores='bell' --grupo helice='15x5' --grupo marco=resto \\
        --excluir '94180|bearing|circlip' --separar helice \\
        --ejes +z +x +y --origen 0.0142 0 0.0070 --caras 80000

--ejes dice qué eje del CAD es el x (frente), el y (izquierda) y el z (arriba)
del modelo, con signo; el resultado debe ser una rotación (regla de la mano
derecha). --origen es el punto del CAD, en metros, que queda en el origen.
"""
import argparse
import collections
import os
import re
import tempfile

import cascadio
import numpy as np
import trimesh


def matriz_ejes(ejes):
    """Rotación que lleva coordenadas del CAD al modelo; ejes = ('+z', '+x', '+y')."""
    R = np.zeros((3, 3))
    for fila, e in enumerate(ejes):
        R[fila, 'xyz'.index(e[-1])] = -1.0 if e[0] == '-' else 1.0
    if not np.isclose(np.linalg.det(R), 1.0):
        raise SystemExit(f'--ejes {" ".join(ejes)} no es una rotación (refleja el modelo)')
    return R


def cargar(step):
    """Lista de (nombre de pieza, malla en coordenadas del CAD, en metros)."""
    with tempfile.TemporaryDirectory() as tmp:
        glb = os.path.join(tmp, 'modelo.glb')
        cascadio.step_to_glb(step, glb, 0.2, 0.3)
        escena = trimesh.load(glb)
    piezas = []
    for nodo in escena.graph.nodes_geometry:
        T, geom = escena.graph[nodo]
        m = escena.geometry[geom].copy()
        m.apply_transform(T)
        piezas.append((nodo, m))
    return piezas


def nombre_corto(nodo):
    return re.sub(r'(_\d+)+$', '', nodo.split('/')[-1])


def listar(step):
    piezas = cargar(step)
    tam = collections.defaultdict(list)
    for nodo, m in piezas:
        tam[nombre_corto(nodo)].append(m.extents)
    print(f'{len(piezas)} piezas (unidades: m)')
    print(f"{'cuántas':>7}  {'tamaño x':>9} {'tamaño y':>9} {'tamaño z':>9}  nombre")
    for n, ex in sorted(tam.items(), key=lambda kv: -len(kv[1])):
        e = ex[0]
        print(f'{len(ex):7d}  {e[0]:9.4f} {e[1]:9.4f} {e[2]:9.4f}  {n}')
    todo = trimesh.util.concatenate([m for _, m in piezas])
    print('Ensamble completo:', np.round(todo.extents, 4), 'm; centro de la caja:',
          np.round(todo.bounding_box.centroid, 4))


def exportar(step, salida, grupos, excluir, separar, ejes, origen, caras):
    R = matriz_ejes(ejes)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ np.asarray(origen)

    asignadas = collections.defaultdict(list)
    resto = next((g for g, patron in grupos if patron == 'resto'), None)
    for nodo, m in cargar(step):
        n = nombre_corto(nodo)
        if excluir and re.search(excluir, n, re.I):
            continue
        destino = next((g for g, patron in grupos
                        if patron != 'resto' and re.search(patron, n, re.I)), resto)
        if destino:
            m.apply_transform(T)
            asignadas[destino].append(m)

    os.makedirs(salida, exist_ok=True)
    for g, mallas in asignadas.items():
        partes = mallas if g in separar else [trimesh.util.concatenate(mallas)]
        for k, m in enumerate(partes):
            nombre = f'{g}_{k}' if g in separar else g
            centro = m.bounding_box.centroid.copy()
            if g in separar:
                m.apply_translation(-centro)
            if caras and len(m.faces) > caras:
                m = m.simplify_quadric_decimation(face_count=caras)
            ruta = os.path.join(salida, f'{nombre}.stl')
            m.export(ruta)
            print(f'{ruta}: {len(m.faces)} caras, tamaño {np.round(m.extents, 4)} m, '
                  f'centro {np.round(centro, 4)} m')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='orden', required=True)
    a = sub.add_parser('listar')
    a.add_argument('step')
    b = sub.add_parser('exportar')
    b.add_argument('step')
    b.add_argument('salida')
    b.add_argument('--grupo', action='append', required=True, help="nombre='regex' o nombre=resto")
    b.add_argument('--excluir', default='', help='regex de piezas que se omiten')
    b.add_argument('--separar', nargs='*', default=[], help='grupos que se exportan pieza por pieza')
    b.add_argument('--ejes', nargs=3, default=['+x', '+y', '+z'], metavar=('FRENTE', 'IZQ', 'ARRIBA'),
                   help='eje del CAD (con signo) que es x, y, z del modelo')
    b.add_argument('--origen', nargs=3, type=float, default=[0, 0, 0], metavar=('X', 'Y', 'Z'))
    b.add_argument('--caras', type=int, default=0, help='máximo de caras por malla (0 = sin reducir)')
    args = ap.parse_args()
    if args.orden == 'listar':
        listar(args.step)
    else:
        grupos = [tuple(g.split('=', 1)) for g in args.grupo]
        exportar(args.step, args.salida, grupos, args.excluir, set(args.separar),
                 args.ejes, args.origen, args.caras)


if __name__ == '__main__':
    main()
