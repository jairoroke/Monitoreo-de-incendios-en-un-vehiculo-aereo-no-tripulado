"""Perturbación de viento: fuerza horizontal acotada con derivada acotada.

    F(t) = F0 + A sin(2 pi f t)     a lo largo de la dirección 'direccion' [rad]

Se publica a partir de t_inicio en /cuadrirotor/viento; el sistema
FuerzaViento de Gazebo la aplica sobre base_link en cada paso de física.
Con sincronizar=True, t_inicio se cuenta desde que llega el aviso de inicio
de misión (/cuadrirotor/mision/inicio), como en la misión con ArduPilot, que
antes espera al EKF y despega.
"""
import math

from geometry_msgs.msg import Vector3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty


class Viento(Node):

    def __init__(self):
        super().__init__('viento')
        p = self.declare_parameters('', [
            ('fuerza_media', 1.0),     # F0 [N]
            ('amplitud', 0.5),         # A [N]
            ('frecuencia', 0.2),       # f [Hz]
            ('direccion', 0.0),        # ángulo en el plano xy [rad]
            ('t_inicio', 15.0),        # [s] de tiempo simulado
            ('periodo', 0.01),         # actualización [s]
            ('sincronizar', False),    # contar t_inicio desde el inicio de misión
        ])
        self.v = {x.name: x.value for x in p}
        self.pub = self.create_publisher(Vector3, '/cuadrirotor/viento', 10)
        self.t0 = None
        if self.v['sincronizar']:
            self.create_subscription(Empty, '/cuadrirotor/mision/inicio', self._inicio, 1)
        self.create_timer(self.v['periodo'], self._paso)

    def _inicio(self, _):
        self.t0 = self.get_clock().now().nanoseconds * 1e-9

    def _paso(self):
        t = self.get_clock().now().nanoseconds * 1e-9
        if t == 0.0:
            return
        if self.t0 is None:
            if self.v['sincronizar']:
                return
            self.t0 = t
        t -= self.t0
        if t < self.v['t_inicio']:
            return
        F = self.v['fuerza_media'] + self.v['amplitud'] * math.sin(
            2 * math.pi * self.v['frecuencia'] * (t - self.v['t_inicio']))
        self.pub.publish(Vector3(x=F * math.cos(self.v['direccion']),
                                 y=F * math.sin(self.v['direccion']), z=0.0))


def main():
    rclpy.init()
    nodo = Viento()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
