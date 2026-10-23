"""Controlador super-twisting adaptativo en el lazo de posición.

Por eje se define la variable deslizante  s = e' + lambda e,  con e = p_d - p.
Con la aceleración comandada  a = a_d + lambda e' - u  se obtiene  s' = u + d,
donde d reúne el viento (F/m), los errores de modelo y el retardo del lazo de
actitud, y u es la salida del algoritmo super-twisting (sta.py) en el esquema
de discretización elegido. El lazo interno es el mismo PD sobre SO(3) de la
línea base, de modo que la comparación aísla el lazo externo.
"""
import numpy as np
import rclpy

from cuadrirotor_sim.controlador_pd import ControladorPD
from cuadrirotor_sim.sta import SuperTwistingAdaptativo


class ControladorSTA(ControladorPD):

    def __init__(self):
        super().__init__('controlador_sta')

    def configurar(self, v):
        p = self.declare_parameters('', [
            ('esquema', 'implicito'),           # explicito | semi_implicito | implicito
            ('lambda', [1.5, 1.5, 2.0]),
            ('k1_0', 1.0), ('eps', 1.0), ('w1', 2.0), ('g1', 1.0),
            ('mu', [0.12, 0.12, 0.02]), ('k1_min', 0.5), ('eta', 0.1), ('k1_max', 8.0),
            ('adaptativo', True),
            ('estimar_perturbacion', False),
        ])
        c = {x.name: x.value for x in p}
        self.lam = np.array(c['lambda'])
        self.esquema = c['esquema']
        self.sta = SuperTwistingAdaptativo(
            3, self.Ts, c['esquema'], k1_0=c['k1_0'], eps=c['eps'], w1=c['w1'], g1=c['g1'],
            mu=np.array(c['mu']), k1_min=c['k1_min'], eta=c['eta'], k1_max=c['k1_max'],
            adaptativo=c['adaptativo'], estimar_perturbacion=c['estimar_perturbacion'])
        self.s = np.zeros(3)

    def descripcion(self):
        return f'Controlador STA adaptativo ({self.esquema})'

    def columnas_extra(self):
        return ['sx', 'sy', 'sz', 'k1x', 'k1y', 'k1z', 'nux', 'nuy', 'nuz']

    def valores_extra(self):
        return [*self.s, *self.sta.k1, *self.sta.nu]

    def aceleracion(self, ep, ev, ad):
        self.s = ev + self.lam * ep
        return ad + self.lam * ev - self.sta(self.s)


def main():
    rclpy.init()
    nodo = ControladorSTA()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
