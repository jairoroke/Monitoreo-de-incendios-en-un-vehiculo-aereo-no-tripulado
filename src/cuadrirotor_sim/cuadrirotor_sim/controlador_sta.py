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
            # Lazo de actitud
            ('actitud', 'pd'),                  # pd | sta
            ('esquema_actitud', 'compensado'),  # explicito | semi_implicito | implicito | compensado
            ('lambda_actitud', [15.0, 15.0, 6.0]),
            ('k1_actitud', 20.0), ('adaptativo_actitud', False),
            ('mu_actitud', [0.05, 0.05, 0.05]), ('k1_max_actitud', 200.0),
            ('w1_actitud', 40.0),
            ('tau_motor', 0.04), ('horizonte_actitud', 8),
            ('retardo_actitud', 0),             # muestras de retardo puro del actuador
        ])
        c = {x.name: x.value for x in p}
        self.lam = np.array(c['lambda'])
        self.esquema = c['esquema']
        self.sta = SuperTwistingAdaptativo(
            3, self.Ts, c['esquema'], k1_0=c['k1_0'], eps=c['eps'], w1=c['w1'], g1=c['g1'],
            mu=np.array(c['mu']), k1_min=c['k1_min'], eta=c['eta'], k1_max=c['k1_max'],
            adaptativo=c['adaptativo'], estimar_perturbacion=c['estimar_perturbacion'])
        self.s = np.zeros(3)
        self.actitud = c['actitud']
        self.esquema_att = c['esquema_actitud']
        self.lam_R = np.array(c['lambda_actitud'])
        self.sta_R = SuperTwistingAdaptativo(
            3, self.Ts, c['esquema_actitud'], k1_0=c['k1_actitud'], eps=1.0,
            w1=c['w1_actitud'], g1=2.0, mu=np.array(c['mu_actitud']), k1_min=1.0, eta=1.0,
            k1_max=c['k1_max_actitud'], adaptativo=c['adaptativo_actitud'],
            tau_actuador=c['tau_motor'], horizonte=c['horizonte_actitud'],
            retardo=c['retardo_actitud'])
        self.s_R = np.zeros(3)

    def descripcion(self):
        att = f'STA {self.esquema_att}' if self.actitud == 'sta' else 'PD'
        return f'Controlador STA adaptativo ({self.esquema}), actitud: {att}'

    def columnas_extra(self):
        return ['sx', 'sy', 'sz', 'k1x', 'k1y', 'k1z', 'nux', 'nuy', 'nuz',
                'sRx', 'sRy', 'sRz', 'k1Rx', 'k1Ry', 'k1Rz']

    def valores_extra(self):
        return [*self.s, *self.sta.k1, *self.sta.nu, *self.s_R, *self.sta_R.k1]

    def par(self, eR, w):
        if self.actitud != 'sta':
            return super().par(eR, w)
        self.s_R = -(w + self.lam_R * eR)
        return -np.diag(self.J) * (self.lam_R * w + self.sta_R(self.s_R))

    def aceleracion(self, ep, ev, ad):
        self.s = ev + self.lam * ep
        return ad + self.lam * ev - self.sta(self.s)

    def aplicado(self, tau, w):
        if self.actitud == 'sta':
            self.sta_R.aplicado(-tau / np.diag(self.J) - self.lam_R * w)


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
