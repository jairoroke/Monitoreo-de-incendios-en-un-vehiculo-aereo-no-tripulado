"""Arma el SDF del mundo 'bosque_grande' con el foco de incendio donde se pida.

El terreno, los árboles, las rocas y los animales los genera una vez
scripts/generar_bosque_grande.py (models/terreno_bosque_grande, con objetos.json
y altura.npz). Aquí solo se escribe el mundo: el launch lo llama al arrancar
para poner el foco en (foco_x, foco_y) sobre el suelo y abrirle un claro (el
área quemada), y el script lo llama para dejar worlds/bosque_grande.sdf con el
foco por omisión.
"""
import json
import math
import os

from cuadrirotor_sim.geo import ORIGEN
from cuadrirotor_sim.terreno import Terreno

RADIO_CLARO = 9.0     # sin árboles ni rocas alrededor del foco [m]

# Malla y medidas de cada especie a escala 1 (copa: radio, altura de la base y de la punta).
ESPECIES = {
    'pino': {'malla': 'model://pino/meshes/pine_tree.dae',
             'copa': 'model://pino/materials/textures/branch_2_diffuse.png',
             'corteza': 'model://pino/materials/textures/bark_diffuse.png',
             'medidas': (1.4, 0.8, 5.1), 'tinte': '0.55 0.68 0.5'},
    'roble': {'malla': 'model://roble/meshes/oak_tree.dae',
              'copa': 'model://roble/materials/textures/branch_diffuse.png',
              'corteza': 'model://roble/materials/textures/bark_diffuse.png',
              'medidas': (4.0, 1.5, 6.5), 'tinte': '1 1 1'},
}


def datos(dir_modelos):
    d = os.path.join(dir_modelos, 'terreno_bosque_grande')
    with open(os.path.join(d, 'objetos.json')) as f:
        return json.load(f), Terreno(os.path.join(d, 'altura.npz'))


def _arbol(i, especie, x, y, z, yaw, s):
    e = ESPECIES[especie]
    r, _, alto = e['medidas']
    pose = f'<pose>{x:.2f} {y:.2f} {z - 0.05:.2f} 0 0 {yaw:.3f}</pose>'
    malla = f"<mesh><uri>{e['malla']}</uri><submesh><name>{{}}</name></submesh><scale>{s:.3f} {s:.3f} {s:.3f}</scale></mesh>"
    return f"""
      <visual name="{especie}_{i}_copa">{pose}
        <geometry>{malla.format('Branch')}</geometry>
        <material><double_sided>true</double_sided><diffuse>{e['tinte']}</diffuse>
          <pbr><metal><albedo_map>{e['copa']}</albedo_map></metal></pbr></material>
      </visual>
      <visual name="{especie}_{i}_tronco">{pose}
        <geometry>{malla.format('Bark')}</geometry>
        <material><diffuse>1 1 1</diffuse>
          <pbr><metal><albedo_map>{e['corteza']}</albedo_map></metal></pbr></material>
      </visual>
      <collision name="{especie}_{i}">
        <pose>{x:.2f} {y:.2f} {z + alto * s / 2:.2f} 0 0 0</pose>
        <geometry><cylinder><radius>{0.12 * r * s:.2f}</radius><length>{alto * s:.2f}</length></cylinder></geometry>
      </collision>"""


def _roca(i, k, x, y, z, roll, pitch, yaw, s, color):
    c = ' '.join(f'{v:.3f}' for v in color)
    malla = f'<mesh><uri>model://terreno_bosque_grande/meshes/roca_{k}.obj</uri><scale>{s:.2f} {s:.2f} {s:.2f}</scale></mesh>'
    pose = f'<pose>{x:.2f} {y:.2f} {z:.2f} {roll:.3f} {pitch:.3f} {yaw:.3f}</pose>'
    return f"""
      <visual name="roca_{i}">{pose}
        <geometry>{malla}</geometry>
        <material><ambient>{c} 1</ambient><diffuse>{c} 1</diffuse><specular>0 0 0 1</specular>
          <pbr><metal><roughness>1.0</roughness><metalness>0.0</metalness></metal></pbr></material>
      </visual>
      <collision name="roca_{i}">{pose}
        <geometry>{malla}</geometry>
      </collision>"""


def sdf(dir_modelos, foco=None):
    obj, ter = datos(dir_modelos)
    fx, fy = foco if foco is not None else obj['foco']
    fz = ter.altura(fx, fy)
    lejos = lambda x, y: math.hypot(x - fx, y - fy) > RADIO_CLARO
    arboles = [a for a in obj['arboles'] if lejos(a[1], a[2])]
    rocas = [r for r in obj['rocas'] if lejos(r[2], r[3])]
    bx, by = obj['base']
    bz = ter.altura(bx, by)
    fauna = ''.join(f"""
    <include>
      <uri>model://fauna_{tipo}</uri>
      <name>{tipo}_{i}</name>
      <pose>{x:.2f} {y:.2f} {ter.altura(x, y):.2f} 0 0 {yaw:.3f}</pose>
    </include>""" for i, (tipo, x, y, yaw) in enumerate(obj['fauna']))
    return f"""<?xml version="1.0"?>
<!-- Generado por cuadrirotor_sim/bosque_grande.py; no editar a mano.
     Valle boscoso de {obj['zona'][1] - obj['zona'][0]:g} x {obj['zona'][3] - obj['zona'][2]:g} m rodeado de montañas: {len(arboles)} árboles,
     {len(rocas)} rocas, {len(obj['fauna'])} animales y el foco de incendio en ({fx:.1f}, {fy:.1f}).
     El vehículo lo agrega el launch en la base ({bx:g}, {by:g}). -->
<sdf version="1.9">
  <world name="bosque_grande">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-particle-emitter-system" name="gz::sim::systems::ParticleEmitter"/>
    <plugin filename="gz-sim-apply-link-wrench-system" name="gz::sim::systems::ApplyLinkWrench"/>

    <gravity>0 0 -9.81</gravity>
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>{ORIGEN[0]}</latitude_deg>
      <longitude_deg>{ORIGEN[1]}</longitude_deg>
      <elevation>{ORIGEN[2]}</elevation>
      <heading_deg>0</heading_deg>
    </spherical_coordinates>
    <scene>
      <ambient>0.5 0.5 0.52 1</ambient>
      <background>0.62 0.74 0.88 1</background>
      <sky></sky>
      <grid>false</grid>
      <shadows>true</shadows>
    </scene>

    <light type="directional" name="sol">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 300 0 0 0</pose>
      <diffuse>0.95 0.92 0.84 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.45 0.35 -0.82</direction>
    </light>

    <include>
      <uri>model://terreno_bosque_grande</uri>
      <name>terreno</name>
      <pose>0 0 0 0 0 0</pose>
    </include>

    <model name="base">
      <static>true</static>
      <pose>{bx:g} {by:g} {bz:.3f} 0 0 0</pose>
      <link name="link">
        <collision name="col">
          <pose>0 0 0.025 0 0 0</pose>
          <geometry><box><size>8 8 0.05</size></box></geometry>
        </collision>
        <visual name="losa">
          <pose>0 0 0.025 0 0 0</pose>
          <geometry><box><size>8 8 0.05</size></box></geometry>
          <material><ambient>0.35 0.35 0.36 1</ambient><diffuse>0.35 0.35 0.36 1</diffuse></material>
        </visual>
        <visual name="h_izq">
          <pose>-1.2 0 0.052 0 0 0</pose>
          <geometry><box><size>0.5 3.6 0.005</size></box></geometry>
          <material><ambient>0.95 0.95 0.95 1</ambient><diffuse>0.95 0.95 0.95 1</diffuse></material>
        </visual>
        <visual name="h_der">
          <pose>1.2 0 0.052 0 0 0</pose>
          <geometry><box><size>0.5 3.6 0.005</size></box></geometry>
          <material><ambient>0.95 0.95 0.95 1</ambient><diffuse>0.95 0.95 0.95 1</diffuse></material>
        </visual>
        <visual name="h_centro">
          <pose>0 0 0.052 0 0 0</pose>
          <geometry><box><size>2.0 0.5 0.005</size></box></geometry>
          <material><ambient>0.95 0.95 0.95 1</ambient><diffuse>0.95 0.95 0.95 1</diffuse></material>
        </visual>
      </link>
    </model>

    <model name="bosque">
      <static>true</static>
      <link name="link">{''.join(_arbol(i, *a) for i, a in enumerate(arboles))}
      </link>
    </model>

    <model name="rocas">
      <static>true</static>
      <link name="link">{''.join(_roca(i, *r) for i, r in enumerate(rocas))}
      </link>
    </model>

    <include>
      <uri>model://foco_incendio</uri>
      <name>foco_incendio</name>
      <pose>{fx:.2f} {fy:.2f} {fz:.2f} 0 0 0</pose>
    </include>
    <include>
      <uri>model://columna_humo</uri>
      <name>columna_humo</name>
      <pose>{fx:.2f} {fy:.2f} {fz:.2f} 0 0 0</pose>
    </include>
{fauna}
  </world>
</sdf>
"""


def escribir(dir_modelos, salida, foco=None):
    with open(salida, 'w') as f:
        f.write(sdf(dir_modelos, foco))
    return salida
