#!/usr/bin/env python3
"""Genera worlds/bosque.sdf: terreno con robles y pinos y un foco de incendio.

Reproduce el escenario del video de ArduPilot (iris_forest: robles y pinos de
OpenRobotics sobre pasto), con una distribución propia y semilla fija para que
el mundo sea siempre el mismo. Deja libres la zona de despegue, el corredor
hacia el foco y un claro alrededor de él, de modo que la misión 'orbita' a la
altura por omisión (3 m) no choca con las copas.

Uso:  python3 scripts/generar_bosque.py worlds/bosque.sdf
"""
import math
import random
import sys

SEMILLA = 7
FOCO = (8.0, 5.0)
RADIO_ORBITA = 4.0
AREA = (-35.0, 45.0, -30.0, 40.0)   # xmin, xmax, ymin, ymax [m]
N_ROBLES, N_PINOS = 50, 35

# Radio de copa de cada especie [m] (medido de las mallas a escala 0.0254).
COPA = {'roble': 5.0, 'pino': 1.6}
HOLGURA = 1.5


def dist_segmento(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def libre(p, especie):
    """Zona de vuelo de la misión: despegue, corredor al foco y anillo de la órbita."""
    r = COPA[especie] + HOLGURA
    inicio_orbita = (FOCO[0] + RADIO_ORBITA, FOCO[1])
    d_foco = math.hypot(p[0] - FOCO[0], p[1] - FOCO[1])
    return (math.hypot(*p) > r + 2.0
            and dist_segmento(p, (0.0, 0.0), inicio_orbita) > r
            and abs(d_foco - RADIO_ORBITA) > r
            and d_foco > 2.0)


def generar():
    rnd = random.Random(SEMILLA)
    arboles = []
    for especie, n in (('roble', N_ROBLES), ('pino', N_PINOS)):
        colocados, intentos = 0, 0
        while colocados < n and intentos < 20000:
            intentos += 1
            p = (rnd.uniform(AREA[0], AREA[1]), rnd.uniform(AREA[2], AREA[3]))
            if not libre(p, especie):
                continue
            if any(math.hypot(p[0] - q[0], p[1] - q[1]) < 0.8 * (COPA[especie] + COPA[e])
                   for e, q, _ in arboles):
                continue
            arboles.append((especie, p, rnd.uniform(0, 2 * math.pi)))
            colocados += 1
    return arboles


def incluir(i, especie, p, yaw):
    return f"""
    <include>
      <uri>model://{especie}</uri>
      <name>{especie}_{i:02d}</name>
      <pose>{p[0]:.2f} {p[1]:.2f} 0 0 0 {yaw:.3f}</pose>
    </include>"""


def main(salida):
    arboles = generar()
    incluidos = ''.join(incluir(i, e, p, yaw) for i, (e, p, yaw) in enumerate(arboles))
    n_r = sum(e == 'roble' for e, *_ in arboles)
    sdf = f"""<?xml version="1.0"?>
<!-- Generado por scripts/generar_bosque.py; no editar a mano.
     {n_r} robles y {len(arboles) - n_r} pinos (OpenRobotics, CC0) y un foco de incendio en
     ({FOCO[0]:g}, {FOCO[1]:g}). El vehículo lo agrega el launch en el origen. -->
<sdf version="1.9">
  <world name="bosque">
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
    <scene>
      <ambient>0.45 0.45 0.45 1</ambient>
      <background>0.62 0.74 0.86 1</background>
      <grid>false</grid>
      <shadows>true</shadows>
    </scene>

    <light type="directional" name="sol">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 50 0 0 0</pose>
      <diffuse>0.95 0.93 0.85 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.3 -0.8</direction>
    </light>

    <model name="terreno">
      <static>true</static>
      <link name="link">
        <collision name="col">
          <geometry><plane><normal>0 0 1</normal><size>300 300</size></plane></geometry>
        </collision>
        <visual name="vis">
          <geometry><plane><normal>0 0 1</normal><size>300 300</size></plane></geometry>
          <material><ambient>0.42 0.55 0.32 1</ambient><diffuse>0.42 0.55 0.32 1</diffuse></material>
        </visual>
      </link>
    </model>

    <include>
      <uri>model://foco_incendio</uri>
      <name>foco_incendio</name>
      <pose>{FOCO[0]:g} {FOCO[1]:g} 0 0 0 0</pose>
    </include>
{incluidos}

  </world>
</sdf>
"""
    with open(salida, 'w') as f:
        f.write(sdf)
    print(f'{salida}: {n_r} robles, {len(arboles) - n_r} pinos')


if __name__ == '__main__':
    main(sys.argv[1])
