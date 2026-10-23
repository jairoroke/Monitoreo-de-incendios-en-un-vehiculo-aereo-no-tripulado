"""Búsqueda del incendio con la cámara y YOLO, y órbita sobre el foco encontrado.

Cadena completa: imagen -> detección (detector_yolo) -> estimación de la
posición del foco (aquí) -> referencia (aquí) -> control (controlador_pd/sta
con mision:=externa). Este nodo nunca usa la posición verdadera del foco: solo
lo que reporta el detector.

Estados:
  despegue      sube en vertical a la altura de búsqueda (sobre las copas).
  patrulla      recorre en zigzag la ruta, con la cámara hacia adelante. Con
                'zona' la ruta son franjas paralelas que cubren todo el bosque
                (separadas 'separacion'); al terminarla la repite, de modo que
                el dron sigue vigilando hasta ver el incendio. Con
                patron:=serpentina la ruta avanza hacia el frente del dron
                (su rumbo al despegar) con barridos de lado a lado unidos por
                semicírculos, y el dron persigue un punto 'anticipo_ruta'
                metros adelante sobre ella (no frena en cada vértice).
  hacia_humo    YOLO confirmó humo: vuela en esa dirección y acumula rumbos.
  aproximacion  hay estimación del foco: va al punto del anillo de órbita más
                cercano, a la altura de búsqueda, mirando al foco.
  verificacion  la estimación salió solo del humo: rodea el punto estimado a la
                altura de búsqueda ('radio_verificacion') mirándolo hasta ver
                las llamas; si no las ve en 'tiempo_verificacion', vuelve a la
                patrulla. Solo con llamas se baja a orbitar y se avisa.
  descenso      baja dentro del claro hasta la altura de órbita.
  orbita        órbita circular con la cámara apuntando al foco estimado.

Terreno: con el parámetro 'terreno' (DEM .npz, mundo 'bosque_grande') las
alturas de búsqueda y de órbita son sobre el suelo. El dron sigue el relieve
mirando 'anticipacion' metros hacia adelante: sube antes de llegar a una loma.
Sin él, el suelo es el plano z = 0.

Evasión de obstáculos (evasion.py): toda consigna que no sea la órbita pasa
por Evasion.corregir con los puntos del LiDAR (/cuadrirotor/lidar/points), que
rodea árboles y rocas; y con 'altura_seguridad' > 0 el dron no avanza mientras
esté a menos de esa altura sobre el relieve que tiene adelante: primero sube.
En la órbita solo se corrige la altura. Los obstáculos que recuerda se publican
en /cuadrirotor/evasion/obstaculos (PointCloud2) para verlos en RViz.

Cobertura: cada 0.5 s se proyecta una rejilla de rayos de la cámara sobre el
terreno (hasta 'alcance_vision') y se marcan las celdas de la zona que ya se
vieron. Se publica el porcentaje y el mapa (nav_msgs/OccupancyGrid, 0 = vista).

Estimación del foco:
  - Con llamas ('fuego'): el rayo de la cámara por el centro de la caja se corta
    con la superficie 1.1 m sobre el suelo (media altura de las llamas), si
    queda a menos de 'alcance_llamas'; se usa la mediana de los últimos cortes.
  - Con humo ('humo'): la columna es vertical sobre el foco, así que cada
    detección da un rumbo; con rumbos desde puntos distintos se triangula por
    mínimos cuadrados.

Aviso con GPS: al empezar la órbita (el foco ya se confirmó con las llamas y se
vio de cerca) se publican las coordenadas del foco una sola vez:
  /cuadrirotor/incendio/gps      sensor_msgs/NavSatFix (QoS transient local:
                                 quien se suscriba después también lo recibe)
  /cuadrirotor/incendio/alerta   std_msgs/String con el texto del aviso
y se agrega una línea a 'registro_incendios' (CSV). 'origen_gps' es la latitud,
longitud y altitud del origen del mundo (ver geo.py). Con 'archivo_mapa' se
escriben ahí (JSON, latitud y longitud) la ruta planeada y el foco confirmado,
que el módulo mavproxy_trayectoria dibuja en el mapa de MAVProxy.
"""
from collections import deque
from datetime import datetime
import json
import math
import os

from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion, Transform, Twist, Vector3
from nav_msgs.msg import OccupancyGrid, Odometry, Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import NavSatFix, NavSatStatus, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Empty, Float32, String
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint
from vision_msgs.msg import Detection2DArray

from cuadrirotor_sim.cobertura import Cobertura, cortar_rayos, rayos_camara, ruta_barrido, ruta_serpentina
from cuadrirotor_sim.dinamica import envolver, quat_a_rot
from cuadrirotor_sim.evasion import Evasion
from cuadrirotor_sim.geo import ORIGEN, enu_a_geodesica, texto_dms
from cuadrirotor_sim.terreno import Terreno

RUTA = [-35.0, -30.0, 45.0, -30.0, 45.0, -10.0, -35.0, -10.0,
        -35.0, 10.0, 45.0, 10.0, 45.0, 30.0, -35.0, 30.0]


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class BusquedaIncendio(Node):

    def __init__(self):
        super().__init__('busqueda_incendio')
        p = self.declare_parameters('', [
            ('altura_busqueda', 15.0), ('altura_orbita', 3.0),
            ('radio', 4.0), ('omega', 0.3),
            ('velocidad', 2.5), ('velocidad_vertical', 1.0), ('aceleracion', 1.5),
            ('ruta', RUTA),
            ('terreno', ''), ('zona', [0.0]), ('separacion', 60.0), ('anticipacion', 40.0),
            ('patron', 'franjas'), ('anticipo_ruta', 10.0),
            ('altura_minima_copas', 0.0), ('alcance_vision', 150.0), ('alcance_llamas', 100.0), ('celda_cobertura', 10.0),
            ('confianza', 0.45), ('confirmar', 3),
            ('camara_xyz', [0.12, 0.0, -0.06]), ('camara_pitch', 0.5236),
            ('camara_hfov', 1.3), ('camara_ancho', 640), ('camara_alto', 480),
            ('evasion', True), ('radio_evasion', 3.0), ('alcance_evasion', 20.0),
            ('altura_seguridad', 0.0), ('lidar_xyz', [0.0, 0.0, 0.1]),
            ('radio_verificacion', 35.0), ('tiempo_verificacion', 120.0),
            ('origen_gps', list(ORIGEN)),
            ('registro_incendios', os.path.expanduser('~/incendios_detectados.csv')),
            ('archivo_mapa', ''),
            ('periodo', 0.02),
        ])
        self.v = {x.name: x.value for x in p}
        self.terreno = Terreno(self.v['terreno'])
        self.evasion = Evasion(self.terreno, radio=self.v['radio_evasion'],
                               alcance=self.v['alcance_evasion'])
        self.desviado = False
        self.n_desvios = 0
        self.fuente = None
        self.zona = self.v['zona'] if len(self.v['zona']) == 4 else None
        self.ruta = np.reshape(self.v['ruta'], (-1, 2))   # con 'zona' se rehace al despegar
        self.cobertura = Cobertura(self.zona or (-50.0, 50.0, -50.0, 50.0), self.v['celda_cobertura'])
        self.avance_reportado = 0
        self.vuelta = 0
        self.f = (self.v['camara_ancho'] / 2) / math.tan(self.v['camara_hfov'] / 2)
        self.Rc = rot_y(self.v['camara_pitch'])

        self.odom = deque(maxlen=400)       # (t, p, R) para emparejar con las imágenes
        self.estado = None
        self.cmd_p = self.cmd_v = None
        self.cmd_yaw = 0.0
        self.i_ruta = 0
        self.racha = {'fuego': 0, 'humo': 0}
        self.cortes_fuego = deque(maxlen=15)
        self.rumbos_humo = deque(maxlen=40)  # (posición xy de la cámara, azimut)
        self.az_humo = None
        self.t_humo = 0.0
        self.foco = None
        self.mapa = {}
        self.entrada = None
        self.t_orbita = None
        self.z_orb = None

        self.pub = self.create_publisher(MultiDOFJointTrajectoryPoint, '/cuadrirotor/consigna', 10)
        self.pub_foco = self.create_publisher(PointStamped, '/cuadrirotor/foco_estimado', 10)
        self.pub_estado = self.create_publisher(String, '/cuadrirotor/busqueda/estado', 10)
        self.pub_inicio = self.create_publisher(Empty, '/cuadrirotor/mision/inicio', 1)
        self.pub_cob = self.create_publisher(Float32, '/cuadrirotor/busqueda/cobertura', 10)
        self.pub_mapa = self.create_publisher(OccupancyGrid, '/cuadrirotor/busqueda/mapa_cobertura', 1)
        self.pub_ruta = self.create_publisher(Path, '/cuadrirotor/busqueda/ruta', 1)
        latched = QoSProfile(depth=5, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_gps = self.create_publisher(NavSatFix, '/cuadrirotor/incendio/gps', latched)
        self.pub_alerta = self.create_publisher(String, '/cuadrirotor/incendio/alerta', latched)
        self.create_subscription(Odometry, '/cuadrirotor/odometria', self._odom, 20)
        self.create_subscription(Detection2DArray, '/cuadrirotor/detecciones', self._detecciones, 10)
        self.pub_obst = self.create_publisher(PointCloud2, '/cuadrirotor/evasion/obstaculos', 1)
        if self.v['evasion']:
            self.create_subscription(PointCloud2, '/cuadrirotor/lidar/points', self._lidar, 5)
            self.create_timer(1.0, self._publicar_obstaculos)
        self.create_timer(self.v['periodo'], self._paso)
        self.create_timer(0.5, self._actualizar_cobertura)
        self.create_timer(2.0, self._publicar_mapa)

    # ------------------------------------------------------------ utilidades
    def _ahora(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _cambiar(self, estado, texto):
        self.estado = estado
        self.get_logger().info(f'[{estado}] {texto}')
        self.pub_estado.publish(String(data=estado))

    def _odom(self, msg):
        o = msg.pose.pose
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.odom.append((t, np.array([o.position.x, o.position.y, o.position.z]),
                          quat_a_rot(o.orientation.x, o.orientation.y, o.orientation.z, o.orientation.w)))

    def _lidar(self, msg):
        if not self.odom:
            return
        q = point_cloud2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        q = q[np.all(np.isfinite(q), axis=1)].astype(float)
        q = q[np.linalg.norm(q, axis=1) < self.v['alcance_evasion'] + 10.0]
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p, R = self._pose_en(t)
        self.evasion.agregar(p + (q + self.v['lidar_xyz']) @ R.T, p)

    def _publicar_obstaculos(self):
        h = PoseStamped().header
        h.stamp = self.get_clock().now().to_msg()
        h.frame_id = 'world'
        self.pub_obst.publish(point_cloud2.create_cloud_xyz32(h, self.evasion.puntos.astype(np.float32)))

    def _pose_en(self, t):
        return min(self.odom, key=lambda e: abs(e[0] - t))[1:]

    def _rayo(self, p, R, u, v):
        """Origen y dirección (mundo) del rayo de la cámara por el píxel (u, v)."""
        W, H = self.v['camara_ancho'], self.v['camara_alto']
        Rcam = R @ self.Rc
        d = Rcam @ np.array([1.0, -(u - W / 2) / self.f, -(v - H / 2) / self.f])
        return p + R @ np.array(self.v['camara_xyz']), d / np.linalg.norm(d)

    # ------------------------------------------------------------- detección
    def _detecciones(self, msg):
        if not self.odom or self.estado in (None, 'despegue', 'descenso', 'orbita'):
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p, R = self._pose_en(t)
        vistos = set()
        for d in msg.detections:
            h = d.results[0].hypothesis
            if h.score < self.v['confianza']:
                continue
            c = d.bbox.center.position
            o, r = self._rayo(p, R, c.x, c.y)
            if h.class_id == 'fuego' and r[2] < -0.05:
                # Lejos, el rayo roza el suelo y un error pequeño de la caja da decenas de metros.
                corte = self.terreno.cortar_rayo(o, r, self.v['alcance_llamas'], desfase=1.1)
                if corte is not None:
                    self.cortes_fuego.append(corte[:2])
            elif h.class_id == 'humo':
                self.rumbos_humo.append((o[:2].copy(), math.atan2(r[1], r[0])))
                self.az_humo = math.atan2(r[1], r[0])
                self.t_humo = self._ahora()
            vistos.add(h.class_id)
        for k in self.racha:
            self.racha[k] = self.racha[k] + 1 if k in vistos else 0
        self._actualizar_estimacion()

    def _actualizar_estimacion(self):
        est = None
        if len(self.cortes_fuego) >= 5:
            est = np.median(np.array(self.cortes_fuego), axis=0)
            fuente = 'llamas'
        elif len(self.rumbos_humo) >= 6:
            az = np.array([a for _, a in self.rumbos_humo])
            if np.ptp(np.unwrap(az)) > 0.25:
                A, b = np.zeros((2, 2)), np.zeros(2)
                for c, a in self.rumbos_humo:
                    n = np.array([math.cos(a), math.sin(a)])
                    P = np.eye(2) - np.outer(n, n)
                    A += P
                    b += P @ c
                est = np.linalg.solve(A, b)
                fuente = 'humo (triangulación)'
        if est is None:
            return
        nuevo = self.foco is None or fuente != self.fuente
        self.foco = est
        self.fuente = fuente
        m = PointStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'world'
        m.point.x, m.point.y, m.point.z = float(est[0]), float(est[1]), 0.0
        self.pub_foco.publish(m)
        if nuevo:
            self.get_logger().info(f'Foco estimado con {fuente}: ({est[0]:.1f}, {est[1]:.1f})')

    # ------------------------------------------------------------ consignas
    def _seguro(self, obj):
        """El objetivo corregido por el relieve (altura_seguridad) y los obstáculos."""
        if self.v['altura_seguridad'] > 0:
            d = obj[:2] - self.cmd_p[:2]
            n = np.linalg.norm(d)
            if n > 1.0:
                b = self.cmd_p[:2] + d / n * min(n, self.v['anticipacion'] / 2)
                z_min = self.terreno.maximo_en_tramo(self.cmd_p[:2], b, margen=5.0) + self.v['altura_seguridad']
                if self.cmd_p[2] < z_min:   # muy bajo para lo que viene: sube sin avanzar
                    return np.array([self.cmd_p[0], self.cmd_p[1], max(obj[2], z_min + 2.0)])
        if not self.v['evasion']:
            return obj
        seguro = self.evasion.corregir(self.cmd_p, obj)
        if self.evasion.activa and not self.desviado:
            self.n_desvios += 1
            self.get_logger().info(f'Evasión: obstáculo en el camino, se desvía (desvío {self.n_desvios})',
                                   throttle_duration_sec=5.0)
        self.desviado = self.evasion.activa
        return seguro

    def _seguir(self, objetivo, yaw_obj, dt):
        """Avanza la consigna hacia el objetivo con velocidad y aceleración limitadas.

        El objetivo pasa antes por _seguro; el error que regresa es respecto al
        objetivo original, para que un desvío no cuente como haber llegado."""
        error = np.linalg.norm(objetivo - self.cmd_p)
        objetivo = self._seguro(objetivo)
        e = objetivo - self.cmd_p
        vmax = np.array([self.v['velocidad']] * 2 + [self.v['velocidad_vertical']])
        amax = self.v['aceleracion']
        v_des = np.sign(e) * np.minimum(vmax, np.sqrt(2 * amax * np.abs(e)))
        exy = np.linalg.norm(e[:2])
        if exy > 1e-6:   # en el plano, avanzar en línea recta hacia el objetivo
            vxy = min(self.v['velocidad'], math.sqrt(2 * amax * exy))
            v_des[:2] = e[:2] / exy * vxy
        a = np.clip((v_des - self.cmd_v) / dt, -amax, amax)
        self.cmd_v = self.cmd_v + a * dt
        self.cmd_p = self.cmd_p + self.cmd_v * dt
        dyaw = envolver(yaw_obj - self.cmd_yaw)
        self.cmd_yaw = envolver(self.cmd_yaw + np.clip(dyaw, -0.6 * dt, 0.6 * dt))
        return a, error

    def _publicar(self, p, v, a, yaw):
        m = MultiDOFJointTrajectoryPoint()
        m.transforms = [Transform(translation=Vector3(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                                  rotation=Quaternion(z=math.sin(yaw / 2), w=math.cos(yaw / 2)))]
        m.velocities = [Twist(linear=Vector3(x=float(v[0]), y=float(v[1]), z=float(v[2])))]
        m.accelerations = [Twist(linear=Vector3(x=float(a[0]), y=float(a[1]), z=float(a[2])))]
        self.pub.publish(m)

    def _z_vuelo(self, destino):
        """Altura de búsqueda sobre el relieve más alto entre la consigna y
        'anticipacion' metros en dirección al destino."""
        a = self.cmd_p[:2]
        d = np.asarray(destino, float) - a
        n = np.linalg.norm(d)
        b = a + d / n * min(n, self.v['anticipacion']) if n > 1e-3 else a
        return self.terreno.maximo_en_tramo(a, b, margen=5.0) + self.v['altura_busqueda']

    def _z_orbita(self):
        """Altura de órbita: sobre el suelo del foco y, si se pide, sobre las copas del anillo."""
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        anillo = self.terreno.altura(self.foco[0] + self.v['radio'] * np.cos(ang),
                                     self.foco[1] + self.v['radio'] * np.sin(ang))
        return max(self.terreno.altura(*self.foco) + self.v['altura_orbita'],
                   float(np.max(anillo)) + self.v['altura_minima_copas'])

    def _preparar_ruta(self, p):
        if self.zona is not None:
            if self.v['patron'] == 'serpentina':
                self.ruta = ruta_serpentina(self.zona, self.v['separacion'], p[:2], self.cmd_yaw)
                avance = self.ruta[-1] - self.ruta[0]
                eje = 'x' if abs(avance[0]) > abs(avance[1]) else 'y'
                signo = '+' if avance[0 if eje == 'x' else 1] > 0 else '-'
                forma = f'serpentina hacia {signo}{eje}, barridos separados {self.v["separacion"]:g} m'
            else:
                self.ruta = ruta_barrido(self.zona, self.v['separacion'], p[:2])
                forma = f'{len(self.ruta) // 2} franjas separadas {self.v["separacion"]:g} m'
            largo = np.sum(np.linalg.norm(np.diff(self.ruta, axis=0), axis=1))
            self.get_logger().info(
                f'Ruta de barrido: {forma}, {largo / 1000:.1f} km (~{largo / self.v["velocidad"] / 60:.0f} min)')
        m = Path()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'world'
        for x, y in self.ruta:
            ps = PoseStamped(header=m.header)
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.position.z = float(self.terreno.altura(x, y) + self.v['altura_busqueda'])
            m.poses.append(ps)
        self.pub_ruta.publish(m)
        self.mapa = {'ruta': [list(enu_a_geodesica(x, y, 0.0, self.v['origen_gps'])[:2])
                              for x, y in self.ruta[::2]]}
        self._escribir_mapa()

    def _escribir_mapa(self):
        ruta = self.v['archivo_mapa']
        if not ruta:
            return
        try:
            tmp = ruta + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(self.mapa, f)
            os.replace(tmp, ruta)
        except OSError as e:
            self.get_logger().error(f'No se pudo escribir {ruta}: {e}')

    def _paso(self):
        if not self.odom:
            return
        dt = self.v['periodo']
        p_dron = self.odom[-1][1]
        if self.estado is None:
            self.cmd_p, self.cmd_v = p_dron.copy(), np.zeros(3)
            R = self.odom[-1][2]
            self.cmd_yaw = math.atan2(R[1, 0], R[0, 0])
            self._preparar_ruta(p_dron)
            self._cambiar('despegue', f"Subiendo a {self.v['altura_busqueda']} m sobre el suelo")

        if self.estado == 'despegue':
            obj = np.array([self.cmd_p[0], self.cmd_p[1], self._z_vuelo(self.ruta[0])])
            a, err = self._seguir(obj, self.cmd_yaw, dt)
            if err < 0.05 and abs(p_dron[2] - obj[2]) < 0.5:
                self._cambiar('patrulla', 'Recorriendo el bosque')
        elif self.estado == 'patrulla':
            wp = self.ruta[self.i_ruta]
            obj = np.array([wp[0], wp[1], self._z_vuelo(wp)])
            yaw = math.atan2(obj[1] - self.cmd_p[1], obj[0] - self.cmd_p[0])
            a, _ = self._seguir(obj, yaw, dt)
            cambio = self.v['anticipo_ruta'] if self.v['patron'] == 'serpentina' else 1.0
            if np.linalg.norm(obj[:2] - self.cmd_p[:2]) < cambio:
                self._siguiente_punto()
            if self.foco is not None:
                self._ir_al_foco()
            elif self.racha['humo'] >= self.v['confirmar']:
                self._cambiar('hacia_humo', f'Humo a la vista, rumbo {math.degrees(self.az_humo):.0f}°')
        elif self.estado == 'hacia_humo':
            destino = self.cmd_p[:2] + 30 * np.array([math.cos(self.az_humo), math.sin(self.az_humo)])
            obj = np.array([*destino, self._z_vuelo(destino)])
            a, _ = self._seguir(obj, self.az_humo, dt)
            if self.foco is not None:
                self._ir_al_foco()
            elif self._ahora() - self.t_humo > 6.0:
                self._cambiar('patrulla', 'Se perdió el humo; sigue el recorrido')
        elif self.estado == 'aproximacion':
            self.entrada = self._punto_entrada()
            obj = np.array([*self.entrada, max(self._z_vuelo(self.entrada), self._z_orbita())])
            yaw = math.atan2(self.foco[1] - self.cmd_p[1], self.foco[0] - self.cmd_p[0])
            a, err = self._seguir(obj, yaw, dt)
            if err < 0.3 and self.fuente != 'llamas':
                self.t_verif = self._ahora()
                self.th_verif = math.atan2(self.cmd_p[1] - self.foco[1], self.cmd_p[0] - self.foco[0])
                self._cambiar('verificacion', 'Estimación solo con humo; rodea el punto hasta ver las llamas')
            elif err < 0.3:
                self._cambiar('descenso', f'Sobre el anillo de órbita; foco en '
                                          f'({self.foco[0]:.1f}, {self.foco[1]:.1f})')
        elif self.estado == 'verificacion':
            Rv = self.v['radio_verificacion']
            self.th_verif += 0.6 * self.v['velocidad'] / Rv * dt
            pt = self.foco + Rv * np.array([math.cos(self.th_verif), math.sin(self.th_verif)])
            obj = np.array([*pt, self._z_vuelo(pt)])
            yaw = math.atan2(self.foco[1] - self.cmd_p[1], self.foco[0] - self.cmd_p[0])
            a, _ = self._seguir(obj, yaw, dt)
            if self.fuente == 'llamas':
                self._ir_al_foco()
            elif self._ahora() - self.t_verif > self.v['tiempo_verificacion']:
                self.foco, self.fuente = None, None
                self.cortes_fuego.clear()
                self.rumbos_humo.clear()
                self._cambiar('patrulla', 'No se vieron llamas; descarta la estimación y sigue el recorrido')
        elif self.estado == 'descenso':
            self.z_orb = self._z_orbita()
            obj = np.array([*self.entrada, self.z_orb])
            yaw = math.atan2(self.foco[1] - self.entrada[1], self.foco[0] - self.entrada[0])
            a, err = self._seguir(obj, yaw, dt)
            if err < 0.05 and abs(envolver(yaw - self.cmd_yaw)) < 0.05:
                self.t_orbita = self._ahora()
                self.z_orb_act = self.z_orb
                self.th0 = math.atan2(self.entrada[1] - self.foco[1], self.entrada[0] - self.foco[0])
                self.pub_inicio.publish(Empty())
                self._avisar_gps()
                self._cambiar('orbita', f'Orbitando el foco a {self.z_orb - self.terreno.altura(*self.foco):.1f} m '
                                        f'sobre el suelo (cobertura {100 * self.cobertura.fraccion():.0f}%)')
        elif self.estado == 'orbita':
            self._orbita()
            return
        self._publicar(self.cmd_p, self.cmd_v, a, self.cmd_yaw)

    def _avisar_gps(self):
        """Publica y registra las coordenadas GPS del foco confirmado."""
        x, y = float(self.foco[0]), float(self.foco[1])
        z = float(self.terreno.altura(x, y))
        lat, lon, alt = enu_a_geodesica(x, y, z, self.v['origen_gps'])
        # Incertidumbre horizontal: dispersión de los cortes de las llamas (mínimo 1 m).
        if len(self.cortes_fuego) >= 5:
            sigma = max(1.0, float(np.std(np.array(self.cortes_fuego), axis=0).max()))
        else:
            sigma = 5.0
        m = NavSatFix()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'world'
        m.status.status = NavSatStatus.STATUS_FIX
        m.status.service = NavSatStatus.SERVICE_GPS
        m.latitude, m.longitude, m.altitude = lat, lon, alt
        m.position_covariance = [sigma ** 2, 0.0, 0.0, 0.0, sigma ** 2, 0.0, 0.0, 0.0, 4 * sigma ** 2]
        m.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        self.pub_gps.publish(m)

        hora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        texto = (f'INCENDIO CONFIRMADO {hora}\n'
                 f'  GPS: {lat:.6f}, {lon:.6f}  ({texto_dms(lat, lon)})\n'
                 f'  Altitud del suelo: {alt:.0f} m   incertidumbre: ±{sigma:.1f} m\n'
                 f'  Mapa: https://www.google.com/maps?q={lat:.6f},{lon:.6f}\n'
                 f'  Local: ({x:.1f}, {y:.1f}) m')
        self.pub_alerta.publish(String(data=texto))
        self.mapa['incendio'] = [lat, lon, alt]
        self._escribir_mapa()
        self.get_logger().warn('\n' + texto)

        ruta = self.v['registro_incendios']
        if ruta:
            try:
                nuevo = not os.path.exists(ruta)
                with open(ruta, 'a') as f:
                    if nuevo:
                        f.write('fecha,latitud,longitud,altitud_m,incertidumbre_m,x_m,y_m\n')
                    f.write(f'{hora},{lat:.7f},{lon:.7f},{alt:.1f},{sigma:.1f},{x:.2f},{y:.2f}\n')
            except OSError as e:
                self.get_logger().error(f'No se pudo escribir {ruta}: {e}')

    def _siguiente_punto(self):
        self.i_ruta += 1
        if self.i_ruta < len(self.ruta):
            return
        self.vuelta += 1
        if self.zona is not None:   # de regreso por las mismas franjas
            self.ruta = self.ruta[::-1]
            self.i_ruta = 1
            self.get_logger().info(f'Recorrido {self.vuelta} completo sin ver el incendio '
                                   f'(cobertura {100 * self.cobertura.fraccion():.0f}%); se repite')
        else:
            self.i_ruta = 0

    def _actualizar_cobertura(self):
        if not self.odom or self.estado is None:
            return
        _, p, R = self.odom[-1]
        o = p + R @ np.array(self.v['camara_xyz'])
        dirs = rayos_camara(R @ self.Rc, self.f, self.v['camara_ancho'], self.v['camara_alto'])
        self.cobertura.marcar(cortar_rayos(self.terreno, o, dirs, self.v['alcance_vision']))
        frac = self.cobertura.fraccion()
        self.pub_cob.publish(Float32(data=100 * frac))
        if self.zona is not None and int(frac * 10) > self.avance_reportado:
            self.avance_reportado = int(frac * 10)
            self.get_logger().info(f'Cobertura del bosque: {100 * frac:.0f}%')

    def _publicar_mapa(self):
        if self.zona is None:
            return
        c = self.cobertura
        m = OccupancyGrid()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'world'
        m.info.resolution = float(c.celda)
        m.info.width, m.info.height = c.nx, c.ny
        m.info.origin.position.x, m.info.origin.position.y = float(c.xmin), float(c.ymin)
        m.info.origin.orientation.w = 1.0
        m.data = np.where(c.vista, 0, -1).astype(np.int8).ravel().tolist()
        self.pub_mapa.publish(m)

    def _ir_al_foco(self):
        self._cambiar('aproximacion', f'Hacia el foco estimado ({self.foco[0]:.1f}, {self.foco[1]:.1f})')

    def _punto_entrada(self):
        """Punto del anillo de órbita más cercano al dron."""
        d = self.cmd_p[:2] - self.foco
        n = np.linalg.norm(d)
        u = d / n if n > 1e-3 else np.array([1.0, 0.0])
        return self.foco + self.v['radio'] * u

    def _orbita(self):
        R, w, Tr = self.v['radio'], self.v['omega'], 4.0
        tau = self._ahora() - self.t_orbita
        if tau < Tr:
            th, dth, ddth = w * tau**2 / (2 * Tr), w * tau / Tr, w / Tr
        else:
            th, dth, ddth = w * (tau - Tr / 2), w, 0.0
        th += self.th0
        c, s = math.cos(th), math.sin(th)
        fx, fy = self.foco
        # Altura: la de órbita, o 'radio_evasion' sobre lo más alto que vea el LiDAR bajo el anillo.
        z_obj = self.z_orb
        if self.v['evasion']:
            z_obj = max(z_obj, self.evasion.techo((fx + R * c, fy + R * s)) + self.v['radio_evasion'])
        dt = self.v['periodo']
        self.z_orb_act += float(np.clip(z_obj - self.z_orb_act, -0.5 * dt, 1.0 * dt))
        p = np.array([fx + R * c, fy + R * s, self.z_orb_act])
        v = np.array([-R * s * dth, R * c * dth, 0.0])
        a = np.array([-R * (c * dth**2 + s * ddth), R * (-s * dth**2 + c * ddth), 0.0])
        self._publicar(p, v, a, th + math.pi)


def main():
    rclpy.init()
    nodo = BusquedaIncendio()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
