"""Mueve los animales del mundo 'bosque_grande' (venados, lobos y zorros).

Cada animal alterna entre quedarse quieto (pastar, olfatear) y caminar hacia un
punto al azar cerca de su claro; los lobos a veces trotan. Va a ras del suelo,
inclinado según la pendiente, y gira poco a poco hacia donde camina. Todas las
poses se mandan juntas a Gazebo con el servicio set_pose_vector a 10 Hz de
tiempo simulado. No usa ROS: solo gz-transport.

Uso (lo lanza simulacion.launch.py con mundo:=bosque_grande):
    python3 -m cuadrirotor_sim.fauna <mundo> <directorio de modelos>
"""
import math
import random
import sys
import threading
import time

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.clock_pb2 import Clock
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node

from cuadrirotor_sim.bosque_grande import datos
from cuadrirotor_sim.terreno import pendiente

# tipo: (velocidad al caminar, velocidad al trotar, prob. de trotar, radio del territorio,
#        pausa mínima y máxima [s])
COMPORTAMIENTO = {
    'venado': (1.0, 1.0, 0.0, 22.0, 6.0, 25.0),
    'lobo': (1.4, 3.2, 0.35, 35.0, 3.0, 12.0),
    'zorro': (0.9, 2.2, 0.2, 18.0, 4.0, 15.0),
}
PERIODO = 0.1


class Animal:

    def __init__(self, nombre, tipo, x, y, yaw, rnd):
        self.nombre, self.tipo, self.rnd = nombre, tipo, rnd
        self.casa = (x, y)
        self.x, self.y, self.yaw = x, y, yaw
        self.v = 0.0
        self.objetivo = None
        self.espera = rnd.uniform(0, 8)

    def paso(self, dt):
        v_cam, v_trote, p_trote, radio, pmin, pmax = COMPORTAMIENTO[self.tipo]
        if self.objetivo is None:
            self.espera -= dt
            if self.espera <= 0:
                a, r = self.rnd.uniform(0, 2 * math.pi), radio * math.sqrt(self.rnd.random())
                self.objetivo = (self.casa[0] + r * math.cos(a), self.casa[1] + r * math.sin(a))
                self.v = v_trote if self.rnd.random() < p_trote else v_cam
            return
        dx, dy = self.objetivo[0] - self.x, self.objetivo[1] - self.y
        d = math.hypot(dx, dy)
        if d < 0.5:
            self.objetivo = None
            self.espera = self.rnd.uniform(pmin, pmax)
            return
        rumbo = math.atan2(dy, dx)
        giro = (rumbo - self.yaw + math.pi) % (2 * math.pi) - math.pi
        self.yaw += max(-1.2 * dt, min(1.2 * dt, giro))
        if abs(giro) < 0.6:      # gira casi en su lugar y luego avanza
            self.x += self.v * dt * math.cos(self.yaw)
            self.y += self.v * dt * math.sin(self.yaw)


def cuaternion(roll, pitch, yaw, q):
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy


def main():
    mundo, modelos = sys.argv[1], sys.argv[2]
    obj, ter = datos(modelos)
    rnd = random.Random(5)
    animales = [Animal(f'{t}_{i}', t, x, y, yaw, rnd) for i, (t, x, y, yaw) in enumerate(obj['fauna'])]

    nodo = Node()
    reloj = {'t': None}
    cv = threading.Condition()

    def al_reloj(msg):
        with cv:
            reloj['t'] = msg.sim.sec + msg.sim.nsec * 1e-9
            cv.notify_all()

    nodo.subscribe(Clock, f'/world/{mundo}/clock', al_reloj)
    servicio = f'/world/{mundo}/set_pose_vector'
    t_prev = None
    while True:
        with cv:
            cv.wait_for(lambda: reloj['t'] is not None and (t_prev is None or reloj['t'] - t_prev >= PERIODO),
                        timeout=5.0)
            t = reloj['t']
        if t is None:
            continue
        if t_prev is None or t < t_prev:     # arranque o reinicio de la simulación
            t_prev = t
            continue
        dt, t_prev = t - t_prev, t
        poses = Pose_V()
        for a in animales:
            a.paso(min(dt, 0.5))
            p = poses.pose.add()
            p.name = a.nombre
            p.position.x, p.position.y = a.x, a.y
            p.position.z = ter.altura(a.x, a.y)
            cuaternion(0.0, -pendiente(ter, a.x, a.y, a.yaw), a.yaw, p.orientation)
        ok, _ = nodo.request(servicio, poses, Pose_V, Boolean, 1000)
        if not ok:
            time.sleep(1.0)


if __name__ == '__main__':
    main()
