"""Controlador base PD en cascada (posición -> actitud) para el cuadrirrotor.

Es la línea base de comparación. El lazo externo calcula la
fuerza deseada F = m (a_d + Kp e_p + Kd e_v + g e3); el interno es un PD sobre
SO(3). El controlador se ejecuta cada Ts segundos de tiempo simulado y manda
velocidades de rotor a Gazebo (retenedor de orden cero).
"""
import csv
import math

from actuator_msgs.msg import Actuators
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint
import numpy as np
import rclpy
from rclpy.node import Node

from cuadrirotor_sim.dinamica import Mezclador, Referencia, quat_a_rot, rot_a_euler, vee


class ControladorPD(Node):

    def __init__(self, nombre='controlador_pd'):
        super().__init__(nombre)
        p = self.declare_parameters('', [
            ('Ts', 0.005),
            ('masa', 1.5), ('g', 9.81),
            ('inercia', [0.0295, 0.0295, 0.056]),
            # Rotor i en (x_i, y_i) [m], marco del cuerpo; sentido +1 ccw, -1 cw.
            ('rotores_xy', [0.225, 0.0, 0.0, 0.225, -0.225, 0.0, 0.0, -0.225]),
            ('rotores_sentido', [1, -1, 1, -1]),
            ('kf', 8.54858e-06), ('km', 0.016), ('w_max', 1000.0),
            ('kp_pos', [4.0, 4.0, 6.0]), ('kd_pos', [3.0, 3.0, 4.0]),
            ('kr_att', [6.6, 6.6, 2.0]), ('kw_att', [0.7, 0.7, 0.6]),
            ('mision', 'hover'), ('altura', 3.0), ('foco', [8.0, 5.0]),
            ('radio', 4.0), ('omega', 0.3),
            ('archivo_csv', ''),
        ])
        v = {x.name: x.value for x in p}
        self.Ts, self.m, self.g = v['Ts'], v['masa'], v['g']
        self.J = np.diag(v['inercia'])
        self.Kp, self.Kd = np.array(v['kp_pos']), np.array(v['kd_pos'])
        self.KR, self.Kw = np.array(v['kr_att']), np.array(v['kw_att'])
        self.mezclador = Mezclador(np.reshape(v['rotores_xy'], (-1, 2)), v['rotores_sentido'],
                                   v['kf'], v['km'], v['w_max'])
        self.args_ref = dict(mision=v['mision'], altura=v['altura'], foco=tuple(v['foco']),
                             radio=v['radio'], omega=v['omega'])

        self.odom = None
        self.n_odom = 0
        self.ref = None
        self.t0 = None

        self.pub_motores = self.create_publisher(Actuators, '/cuadrirotor/motores', 10)
        self.pub_ref = self.create_publisher(PoseStamped, '/cuadrirotor/referencia', 10)
        self.pub_err = self.create_publisher(Vector3Stamped, '/cuadrirotor/error', 10)
        self.create_subscription(Odometry, '/cuadrirotor/odometria', self._odom, 10)
        # mision = 'externa': la referencia llega de otro nodo (p. ej. busqueda_incendio).
        self.consigna = None
        if v['mision'] == 'externa':
            self.create_subscription(MultiDOFJointTrajectoryPoint, '/cuadrirotor/consigna',
                                     self._consigna, 10)
        self.create_timer(self.Ts, self._paso)

        self.configurar(v)

        self.csv = None
        if v['archivo_csv']:
            self.archivo = open(v['archivo_csv'], 'w', newline='')
            self.csv = csv.writer(self.archivo)
            self.csv.writerow(['t', 'x', 'y', 'z', 'xd', 'yd', 'zd', 'phi', 'theta', 'psi',
                               'psid', 'T', 'tx', 'ty', 'tz', 'w0', 'w1', 'w2', 'w3']
                              + self.columnas_extra())
        self.get_logger().info(
            f"{self.descripcion()} listo: Ts = {self.Ts} s, misión = {v['mision']}")

    # Puntos de extensión para otros lazos externos (ver controlador_sta.py).
    def configurar(self, v):
        pass

    def descripcion(self):
        return 'Controlador PD'

    def columnas_extra(self):
        return []

    def valores_extra(self):
        return []

    def aceleracion(self, ep, ev, ad):
        """Aceleración deseada del lazo externo (sin gravedad)."""
        return ad + self.Kp * ep + self.Kd * ev

    def _consigna(self, msg):
        tr, ve, ac = msg.transforms[0], msg.velocities[0], msg.accelerations[0]
        q = tr.rotation
        self.consigna = (np.array([tr.translation.x, tr.translation.y, tr.translation.z]),
                         np.array([ve.linear.x, ve.linear.y, ve.linear.z]),
                         np.array([ac.linear.x, ac.linear.y, ac.linear.z]),
                         2.0 * math.atan2(q.z, q.w))

    def _odom(self, msg):
        # Gazebo promedia la velocidad en 10 muestras y la primera se calcula contra
        # el origen: se descartan los primeros mensajes para no arrastrar el pico.
        self.n_odom += 1
        if self.n_odom > 20:
            self.odom = msg

    def _paso(self):
        if self.odom is None:
            return
        t_sim = self.get_clock().now().nanoseconds * 1e-9
        o = self.odom.pose.pose
        p = np.array([o.position.x, o.position.y, o.position.z])
        R = quat_a_rot(o.orientation.x, o.orientation.y, o.orientation.z, o.orientation.w)
        tw = self.odom.twist.twist
        v = R @ np.array([tw.linear.x, tw.linear.y, tw.linear.z])   # cuerpo -> mundo
        w = np.array([tw.angular.x, tw.angular.y, tw.angular.z])     # marco del cuerpo

        externa = self.args_ref['mision'] == 'externa'
        if externa:
            if self.consigna is None:
                return   # en tierra, motores detenidos, hasta la primera consigna
            if self.t0 is None:
                self.t0 = t_sim
            pd, vd, ad, psid = self.consigna
        elif self.ref is None:
            self.t0 = t_sim
            self.ref = Referencia(p, **self.args_ref)
        t = t_sim - self.t0
        if not externa:
            pd, vd, ad, psid = self.ref(t)

        # Lazo externo: fuerza deseada en el marco inercial.
        ep, ev = pd - p, vd - v
        F = self.m * (self.aceleracion(ep, ev, ad) + np.array([0.0, 0.0, self.g]))
        T = float(F @ R[:, 2])

        # Actitud deseada a partir de la dirección de F y la guiñada deseada.
        b3 = F / np.linalg.norm(F)
        b1c = np.array([math.cos(psid), math.sin(psid), 0.0])
        b2 = np.cross(b3, b1c)
        b2 /= np.linalg.norm(b2)
        Rd = np.column_stack([np.cross(b2, b3), b2, b3])

        # Lazo interno: PD sobre SO(3).
        eR = 0.5 * vee(Rd.T @ R - R.T @ Rd)
        tau = -self.KR * eR - self.Kw * w + np.cross(w, self.J @ w)

        wr = self.mezclador.velocidades(T, tau)
        cmd = Actuators()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.velocity = [float(x) for x in wr]
        self.pub_motores.publish(cmd)

        ref = PoseStamped()
        ref.header.stamp, ref.header.frame_id = cmd.header.stamp, 'world'
        ref.pose.position.x, ref.pose.position.y, ref.pose.position.z = (float(x) for x in pd)
        ref.pose.orientation.z, ref.pose.orientation.w = math.sin(psid / 2), math.cos(psid / 2)
        self.pub_ref.publish(ref)
        err = Vector3Stamped()
        err.header = ref.header
        err.vector.x, err.vector.y, err.vector.z = (float(x) for x in ep)
        self.pub_err.publish(err)

        if self.csv:
            phi, th, psi = rot_a_euler(R)
            self.csv.writerow([f'{t:.4f}', *p, *pd, phi, th, psi, psid, T, *tau, *wr,
                               *self.valores_extra()])

    def destroy_node(self):
        if self.csv:
            self.archivo.close()
        super().destroy_node()


def main():
    rclpy.init()
    nodo = ControladorPD()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
