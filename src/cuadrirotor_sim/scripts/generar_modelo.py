#!/usr/bin/env python3
"""Genera los modelos de Gazebo y sus parámetros para el controlador.

Para cada vehículo escribe:
  models/<vehiculo>/model.sdf          planta que simula Gazebo
  config/vehiculo_<vehiculo>.yaml      los mismos valores para el controlador

Así el modelo Newton-Euler del controlador coincide siempre con la planta.
Todos los modelos se llaman 'cuadrirotor' dentro de Gazebo, de modo que los
tópicos y el puente no cambian al cambiar de vehículo.

Uso:  python3 scripts/generar_modelo.py            (desde el paquete; todos)
      python3 scripts/generar_modelo.py tarot650   (solo uno)
"""
import os
import sys

import numpy as np

# Cámara frontal, igual en todos los vehículos (la usa el detector YOLO).
CAMARA_PITCH = 0.5236   # 30 grados hacia abajo
CAMARA_HFOV = 1.3       # campo de visión horizontal [rad]
CAMARA_ANCHO, CAMARA_ALTO = 640, 480

# Rotores: (x, y, sentido) en el marco del cuerpo; sentido 'ccw' o 'cw'.
VEHICULOS = {
    'cuadrirotor': dict(
        descripcion='Cuadrirrotor genérico en configuración "+" (1.5 kg)',
        masa_cuerpo=1.48,
        inercia_cuerpo=(0.029, 0.029, 0.055),
        rotores=[(0.225, 0.0, 'ccw'), (0.0, 0.225, 'cw'), (-0.225, 0.0, 'ccw'), (0.0, -0.225, 'cw')],
        z_rotor=0.04,
        masa_rotor=0.005,
        inercia_rotor=(9.75e-07, 4.17041e-05, 4.26041e-05),
        radio_helice=0.12,
        kf=8.54858e-06, km=0.016, w_max=1000.0,
        tau_subida=0.0125, tau_bajada=0.025,
        kr_att=[6.6, 6.6, 2.0], kw_att=[0.7, 0.7, 0.6],
        # Adaptación del STA: mu debe superar el ciclo límite que induce el
        # retardo de actitud y motores (|s| ~ 0.05-0.1 m/s en x, y).
        sta=dict(mu=[0.12, 0.12, 0.02], k1_max=8.0),
        z_inicial=0.05,
        camara=(0.1, 0.0, -0.03),
    ),
    # Tarot Ironman 650 con motores T-Motor MN3508 y hélices de 15". La geometría
    # (posición de los motores, tren de aterrizaje) se midió en el CAD de GrabCAD;
    # la masa, las inercias y las constantes de hélice son valores típicos
    # supuestos para esta configuración (ver README).
    'tarot650': dict(
        descripcion='Tarot Ironman 650 en configuración "X" (2.2 kg, MN3508, hélices 15")',
        masa_cuerpo=2.12,
        inercia_cuerpo=(0.0304, 0.0321, 0.0585),
        rotores=[(0.2198, -0.212, 'ccw'), (0.2198, 0.212, 'cw'),
                 (-0.2198, 0.212, 'ccw'), (-0.2198, -0.212, 'cw')],
        z_rotor=0.04,
        masa_rotor=0.02,
        inercia_rotor=(2.0e-07, 2.4e-04, 2.4e-04),
        radio_helice=0.1905,
        kf=4.24e-05, km=0.016, w_max=680.0,
        tau_subida=0.03, tau_bajada=0.05,
        kr_att=[3.4, 3.6, 1.65], kw_att=[0.55, 0.58, 0.59],
        # Motores y hélices más lentos: el ciclo límite en x, y es mayor
        # (|s| ~ 0.2-0.35 m/s con k1 de 0.5 a 2).
        sta=dict(mu=[0.3, 0.3, 0.03], k1_max=3.0),
        ardupilot=True,
        z_inicial=0.235,
        camara=(0.12, 0.0, -0.06),
        # Mallas del CAD ya en el marco del cuerpo (scripts/cad_a_malla.py con
        # --ejes +z +x +y --origen 0.0142 0 0.0070): el frente es paralelo a los
        # patines del tren de aterrizaje; helice_i.stl va en el rotor i.
        # Grupos (scripts/cad_a_malla.py, ver README): placas y brazos sin
        # simplificar; las placas de 1.6 mm se deshacen si se reducen.
        mallas=[('placas', '0.09 0.09 0.1 1'), ('brazos', '0.06 0.06 0.07 1'),
                ('acentos', '0.9 0.42 0.08 1'), ('tren', '0.16 0.16 0.17 1'),
                ('motores', '0.72 0.72 0.75 1'), ('marco', '0.6 0.6 0.62 1')],
        helices_cad=True,
        # Tren de aterrizaje: dos patines a lo largo de x, 0.2205 m bajo el origen.
        patas=[(-0.1065, -0.2205), (0.1065, -0.2205)],
    ),
}


def inercia_total(v):
    """Inercia del vehículo completo (cuerpo + rotores como masas puntuales)."""
    m_r = v['masa_rotor']
    I = np.array(v['inercia_cuerpo'], dtype=float)
    for x, y, _ in v['rotores']:
        I += m_r * np.array([y**2 + v['z_rotor']**2, x**2 + v['z_rotor']**2, x**2 + y**2])
    return [round(float(i), 5) for i in I]


def sdf_vehiculo(nombre, v):
    ixx_r, iyy_r, izz_r = v['inercia_rotor']
    links, joints, plugins = [], [], []
    for i, (x, y, d) in enumerate(v['rotores']):
        color = '0.9 0.35 0.05 1' if x > 0 else '0.1 0.1 0.1 1'   # delanteros en naranja
        n = f'rotor_{i}'
        helice = (f'<mesh><uri>model://{nombre}/meshes/helice_{i}.stl</uri></mesh>' if v.get('helices_cad') else
                  f"<box><size>{2 * v['radio_helice']} 0.028 0.004</size></box>")
        links.append(f"""
    <link name="{n}">
      <pose>{x} {y} {v['z_rotor']} 0 0 0</pose>
      <inertial>
        <mass>{v['masa_rotor']}</mass>
        <inertia>
          <ixx>{ixx_r}</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>{iyy_r}</iyy><iyz>0</iyz><izz>{izz_r}</izz>
        </inertia>
      </inertial>
      <visual name="helice">
        <geometry>{helice}</geometry>
        <material><ambient>{color}</ambient><diffuse>{color}</diffuse></material>
      </visual>
    </link>""")
        joints.append(f"""
    <joint name="{n}_joint" type="revolute">
      <parent>base_link</parent>
      <child>{n}</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit><lower>-1e16</lower><upper>1e16</upper></limit>
      </axis>
    </joint>""")
        plugins.append(f"""
    <plugin filename="gz-sim-multicopter-motor-model-system"
            name="gz::sim::systems::MulticopterMotorModel">
      <robotNamespace>cuadrirotor</robotNamespace>
      <jointName>{n}_joint</jointName>
      <linkName>{n}</linkName>
      <turningDirection>{d}</turningDirection>
      <timeConstantUp>{v['tau_subida']}</timeConstantUp>
      <timeConstantDown>{v['tau_bajada']}</timeConstantDown>
      <maxRotVelocity>{v['w_max']}</maxRotVelocity>
      <motorConstant>{v['kf']}</motorConstant>
      <momentConstant>{v['km']}</momentConstant>
      <commandSubTopic>command/motor_speed</commandSubTopic>
      <motorNumber>{i}</motorNumber>
      <rotorDragCoefficient>8.06428e-05</rotorDragCoefficient>
      <rollingMomentCoefficient>1e-06</rollingMomentCoefficient>
      <motorSpeedPubTopic>motor_speed/{i}</motorSpeedPubTopic>
      <rotorVelocitySlowdownSim>10</rotorVelocitySlowdownSim>
      <motorType>velocity</motorType>
    </plugin>""")

    if 'mallas' in v:
        cuerpo = ''.join(f"""
      <visual name="{m}">
        <geometry><mesh><uri>model://{nombre}/meshes/{m}.stl</uri></mesh></geometry>
        <material><ambient>{c}</ambient><diffuse>{c}</diffuse><specular>0.3 0.3 0.3 1</specular></material>
      </visual>""" for m, c in v['mallas'])
        colisiones = """
      <collision name="cuerpo_col">
        <geometry><box><size>0.16 0.16 0.06</size></box></geometry>
      </collision>""" + ''.join(f"""
      <collision name="pata_{k}">
        <pose>0 {y} {z} 0 1.5707963 0</pose>
        <geometry><cylinder><radius>0.005</radius><length>0.2</length></cylinder></geometry>
      </collision>""" for k, (y, z) in enumerate(v['patas']))
    else:
        l = max(abs(x) + abs(y) for x, y, _ in v['rotores'])
        cuerpo = """
      <visual name="cuerpo">
        <geometry><box><size>0.16 0.16 0.05</size></box></geometry>
        <material><ambient>0.15 0.35 0.7 1</ambient><diffuse>0.15 0.35 0.7 1</diffuse></material>
      </visual>""" + ''.join(f"""
      <visual name="brazo_{k}">
        <pose>0 0 0 0 0 {yaw}</pose>
        <geometry><box><size>{2 * l} 0.025 0.02</size></box></geometry>
        <material><ambient>0.2 0.2 0.25 1</ambient><diffuse>0.2 0.2 0.25 1</diffuse></material>
      </visual>""" for k, yaw in ((0, 0), (1, 1.5707963)))
        colisiones = """
      <collision name="cuerpo_col">
        <geometry><box><size>0.2 0.2 0.06</size></box></geometry>
      </collision>"""

    ixx, iyy, izz = v['inercia_cuerpo']
    cx, cy, cz = v['camara']
    return f"""<?xml version="1.0"?>
<!-- Generado por scripts/generar_modelo.py; no editar a mano.
     {v['descripcion']}.
     kf = {v['kf']} N/(rad/s)^2, km = {v['km']} m, w_max = {v['w_max']} rad/s. -->
<sdf version="1.9">
  <model name="{nombre}">
    <link name="base_link">
      <inertial>
        <mass>{v['masa_cuerpo']}</mass>
        <inertia>
          <ixx>{ixx}</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>{iyy}</iyy><iyz>0</iyz><izz>{izz}</izz>
        </inertia>
      </inertial>{colisiones}{cuerpo}

      <sensor name="imu" type="imu">
        <always_on>1</always_on>
        <update_rate>200</update_rate>
        <topic>cuadrirotor/imu</topic>
      </sensor>

      <!-- Cámara frontal inclinada 30 grados hacia abajo: ve el horizonte para
           buscar el incendio de lejos y lo mantiene a la vista en la órbita. -->
      <sensor name="camara" type="camera">
        <pose>{cx} {cy} {cz} 0 {CAMARA_PITCH} 0</pose>
        <always_on>1</always_on>
        <update_rate>15</update_rate>
        <topic>cuadrirotor/camara</topic>
        <camera>
          <horizontal_fov>{CAMARA_HFOV}</horizontal_fov>
          <image><width>{CAMARA_ANCHO}</width><height>{CAMARA_ALTO}</height><format>R8G8B8</format></image>
          <clip><near>0.1</near><far>1500</far></clip>
        </camera>
      </sensor>

      <!-- LiDAR 3D para la evasión de obstáculos (busqueda_incendio): 360 grados
           en 120 haces por 16 canales de -35 a +15 grados, hasta 30 m. -->
      <sensor name="lidar" type="gpu_lidar">
        <pose>0 0 0.1 0 0 0</pose>
        <always_on>1</always_on>
        <update_rate>10</update_rate>
        <topic>cuadrirotor/lidar</topic>
        <lidar>
          <scan>
            <horizontal><samples>120</samples><min_angle>-3.14159</min_angle><max_angle>3.14159</max_angle></horizontal>
            <vertical><samples>16</samples><min_angle>-0.6109</min_angle><max_angle>0.2618</max_angle></vertical>
          </scan>
          <range><min>0.8</min><max>30</max><resolution>0.05</resolution></range>
        </lidar>
      </sensor>
    </link>
{''.join(links)}
{''.join(joints)}
{''.join(plugins)}

    <plugin filename="gz-sim-odometry-publisher-system"
            name="gz::sim::systems::OdometryPublisher">
      <dimensions>3</dimensions>
      <odom_publish_frequency>1000</odom_publish_frequency>
      <odom_frame>world</odom_frame>
      <robot_base_frame>base_link</robot_base_frame>
    </plugin>

    <!-- Perturbación externa en el marco inercial (paquete cuadrirotor_gz_plugins). -->
    <plugin filename="FuerzaViento" name="cuadrirotor::FuerzaViento">
      <link_name>base_link</link_name>
      <topic>/cuadrirotor/viento</topic>
    </plugin>
  </model>
</sdf>
"""


def yaml_vehiculo(nombre, v):
    masa = round(v['masa_cuerpo'] + len(v['rotores']) * v['masa_rotor'], 4)
    xy = [c for x, y, _ in v['rotores'] for c in (x, y)]
    sentidos = [1 if d == 'ccw' else -1 for *_, d in v['rotores']]
    return f"""# Generado por scripts/generar_modelo.py a partir del modelo '{nombre}'; no editar a mano.
# {v['descripcion']}.
/**:
  ros__parameters:
    masa: {masa}
    inercia: {inercia_total(v)}
    rotores_xy: {xy}
    rotores_sentido: {sentidos}
    kf: {v['kf']}
    km: {v['km']}
    w_max: {v['w_max']}
    # Lazo interno PD sobre SO(3): ganancias proporcionales a la inercia.
    kr_att: {v['kr_att']}
    kw_att: {v['kw_att']}
    # Altura de aparición en Gazebo [m].
    z_inicial: {v['z_inicial']}
    # Cámara frontal: posición en el cuerpo [m], inclinación hacia abajo [rad],
    # campo de visión horizontal [rad] y resolución.
    camara_xyz: {list(v['camara'])}
    camara_pitch: {CAMARA_PITCH}
    camara_hfov: {CAMARA_HFOV}
    camara_ancho: {CAMARA_ANCHO}
    camara_alto: {CAMARA_ALTO}

controlador_sta:
  ros__parameters:
    mu: {v['sta']['mu']}
    k1_max: {v['sta']['k1_max']}
"""


# Orden de motores de ArduPilot para un quad "X": M1 adelante-derecha (ccw),
# M2 atrás-izquierda (ccw), M3 adelante-izquierda (cw), M4 atrás-derecha (cw).
# Se busca, para cada salida de ArduPilot, el rotor del modelo en esa esquina.
ESQUINAS_ARDUPILOT = [(1, -1), (-1, 1), (1, 1), (-1, -1)]


def sdf_ardupilot(nombre, v):
    """Variante del vehículo pilotada por ArduPilot SITL (plugin ardupilot_gazebo).

    ArduPilotPlugin publica cada salida de motor (PWM 1000-2000 -> 0-w_max
    rad/s) en un tópico; ComandosMotores las junta para MulticopterMotorModel,
    de modo que la física de los motores es la misma que con los controladores
    de ROS 2.
    """
    sdf = sdf_vehiculo(nombre, v)
    n_rot = len(v['rotores'])
    signos = [(np.sign(x), np.sign(y)) for x, y, _ in v['rotores']]
    controles = []
    for canal, esquina in enumerate(ESQUINAS_ARDUPILOT[:n_rot]):
        r = signos.index(esquina)
        controles.append(f"""
      <control channel="{canal}">
        <jointName>rotor_{r}_joint</jointName>
        <type>COMMAND</type>
        <cmd_topic>/cuadrirotor/ap/motor_{r}</cmd_topic>
        <useForce>0</useForce>
        <multiplier>{v['w_max']}</multiplier>
        <offset>0</offset>
        <servo_min>1000</servo_min>
        <servo_max>2000</servo_max>
      </control>""")
    entradas = ''.join(f"""
      <entrada>/cuadrirotor/ap/motor_{r}</entrada>""" for r in range(n_rot))
    imu = """
      <!-- IMU que lee ArduPilot (marco NED: girada 180 grados en x). -->
      <sensor name="imu_sensor" type="imu">
        <pose degrees="true">0 0 0 180 0 0</pose>
        <always_on>1</always_on>
        <update_rate>1000</update_rate>
      </sensor>
    </link>"""
    plugins = f"""
    <plugin filename="ArduPilotPlugin" name="ArduPilotPlugin">
      <fdm_addr>127.0.0.1</fdm_addr>
      <fdm_port_in>9002</fdm_port_in>
      <connectionTimeoutMaxCount>5</connectionTimeoutMaxCount>
      <lock_step>1</lock_step>
      <no_time_sync>1</no_time_sync>
      <have_32_channels>0</have_32_channels>
      <modelXYZToAirplaneXForwardZDown degrees="true">0 0 0 180 0 0</modelXYZToAirplaneXForwardZDown>
      <gazeboXYZToNED degrees="true">0 0 0 180 0 90</gazeboXYZToNED>
      <imuName>base_link::imu_sensor</imuName>{''.join(controles)}
    </plugin>

    <plugin filename="ComandosMotores" name="cuadrirotor::ComandosMotores">{entradas}
      <salida>/cuadrirotor/command/motor_speed</salida>
    </plugin>
  </model>"""
    sdf = sdf.replace('    </link>', imu, 1)
    sdf = sdf.replace('  </model>', plugins, 1)
    return sdf.replace(f'<model name="{nombre}">', f'<model name="{nombre}_ardupilot">', 1)


def main(nombres):
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for n in nombres:
        v = VEHICULOS[n]
        os.makedirs(os.path.join(raiz, 'models', n), exist_ok=True)
        with open(os.path.join(raiz, 'models', n, 'model.sdf'), 'w') as f:
            f.write(sdf_vehiculo(n, v))
        with open(os.path.join(raiz, 'config', f'vehiculo_{n}.yaml'), 'w') as f:
            f.write(yaml_vehiculo(n, v))
        print(f'{n}: models/{n}/model.sdf, config/vehiculo_{n}.yaml; inercia total {inercia_total(v)}')
        if v.get('ardupilot'):
            d = os.path.join(raiz, 'models', f'{n}_ardupilot')
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, 'model.sdf'), 'w') as f:
                f.write(sdf_ardupilot(n, v))
            with open(os.path.join(d, 'model.config'), 'w') as f:
                f.write(f'<?xml version="1.0"?>\n<model>\n  <name>{n}_ardupilot</name>\n'
                        f'  <version>1.0</version>\n  <sdf version="1.9">model.sdf</sdf>\n'
                        f'  <description>{v["descripcion"]}, pilotado por ArduPilot SITL.</description>\n'
                        f'</model>\n')
            print(f'{n}: models/{n}_ardupilot/model.sdf (ArduPilot SITL)')


if __name__ == '__main__':
    main(sys.argv[1:] or list(VEHICULOS))
