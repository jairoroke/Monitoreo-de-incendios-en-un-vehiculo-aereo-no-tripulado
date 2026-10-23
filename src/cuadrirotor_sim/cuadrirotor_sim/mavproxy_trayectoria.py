"""Módulo de MAVProxy: dibuja en el mapa la trayectoria recorrida por el dron.

  - Trayectoria recorrida (magenta): todas las posiciones GPS del dron desde que
    despega, como una línea continua (un punto cada 'paso' metros).
  - Ruta planeada (cian tenue) y foco del incendio confirmado (círculo rojo):
    los escribe busqueda_incendio en un JSON ('archivo_mapa'); este módulo lo
    vuelve a leer cuando cambia.

Se carga con  mavproxy.py --load-module=cuadrirotor_sim.mavproxy_trayectoria
(el launch lo hace con controlador:=ardupilot y GUI). Comandos en el prompt:

  trayectoria borrar    borra la trayectoria recorrida
  trayectoria ocultar   / trayectoria mostrar
"""
import json
import math
import os
import tempfile
import time

from MAVProxy.modules.lib import mp_module

ARCHIVO = os.environ.get('CUADRIROTOR_MAPA', os.path.join(
    tempfile.gettempdir(), f'cuadrirotor_sim_{os.getuid()}', 'mapa_busqueda.json'))
CAPA = 'Trayectoria'


def distancia(a, b):
    """Distancia horizontal aproximada en metros entre dos (lat, lon)."""
    dy = (b[0] - a[0]) * 111_320.0
    dx = (b[1] - a[1]) * 111_320.0 * math.cos(math.radians(a[0]))
    return math.hypot(dx, dy)


class Trayectoria(mp_module.MPModule):

    def __init__(self, mpstate):
        super().__init__(mpstate, 'trayectoria', 'trayectoria recorrida en el mapa')
        self.puntos = []
        self.paso = 2.0
        self.visible = True
        self.cambio = False
        self.t_dibujo = 0.0
        self.t_archivo = 0.0
        self.mtime = None
        self.datos = {}
        self.add_command('trayectoria', self.comando, 'trayectoria recorrida en el mapa',
                         ['borrar', 'ocultar', 'mostrar'])

    def comando(self, args):
        if not args:
            print(f'trayectoria: {len(self.puntos)} puntos; uso: trayectoria borrar|ocultar|mostrar')
        elif args[0] == 'borrar':
            self.puntos = []
            self.cambio = True
        elif args[0] in ('ocultar', 'mostrar'):
            self.visible = args[0] == 'mostrar'
            self.cambio = True
            self.mtime = None

    def mavlink_packet(self, m):
        if m.get_type() != 'GLOBAL_POSITION_INT' or m.relative_alt < 1000:   # en el suelo
            return
        p = (m.lat * 1e-7, m.lon * 1e-7)
        if not self.puntos or distancia(self.puntos[-1], p) >= self.paso:
            self.puntos.append(p)
            self.cambio = True

    def _mapa(self):
        mod = self.module('map')
        return getattr(mod, 'map', None) if mod else None

    def idle_task(self):
        ahora = time.time()
        mapa = self._mapa()
        if mapa is None:
            return
        from MAVProxy.modules.mavproxy_map import mp_slipmap
        if ahora - self.t_archivo > 2.0:
            self.t_archivo = ahora
            self._leer_archivo(mapa, mp_slipmap)
        if self.cambio and ahora - self.t_dibujo > 1.0 and len(self.puntos) >= 2:
            self.t_dibujo = ahora
            self.cambio = False
            if self.visible:
                mapa.add_object(mp_slipmap.SlipPolygon('trayectoria_recorrida', list(self.puntos), layer=CAPA,
                                                       colour=(255, 0, 255), linewidth=2, showcircles=False))
            else:
                mapa.add_object(mp_slipmap.SlipRemoveObject('trayectoria_recorrida'))

    def _leer_archivo(self, mapa, mp_slipmap):
        try:
            mtime = os.path.getmtime(ARCHIVO)
        except OSError:
            return
        if mtime == self.mtime:
            return
        self.mtime = mtime
        try:
            with open(ARCHIVO) as f:
                self.datos = json.load(f)
        except (OSError, ValueError):
            self.mtime = None     # a medio escribir; se reintenta
            return
        ruta = [tuple(p) for p in self.datos.get('ruta', [])]
        if self.visible and len(ruta) >= 2:
            mapa.add_object(mp_slipmap.SlipPolygon('ruta_planeada', ruta, layer=CAPA,
                                                   colour=(0, 160, 160), linewidth=1, showcircles=False))
        else:
            mapa.add_object(mp_slipmap.SlipRemoveObject('ruta_planeada'))
        foco = self.datos.get('incendio')
        if self.visible and foco:
            mapa.add_object(mp_slipmap.SlipCircle('incendio', CAPA, tuple(foco[:2]), 15.0,
                                                  (255, 0, 0), 3))
            mapa.add_object(mp_slipmap.SlipLabel('incendio_texto', tuple(foco[:2]),
                                                 f'INCENDIO {foco[0]:.6f}, {foco[1]:.6f}', CAPA,
                                                 (255, 60, 60), size=0.6))
        else:
            mapa.add_object(mp_slipmap.SlipRemoveObject('incendio'))
            mapa.add_object(mp_slipmap.SlipRemoveObject('incendio_texto'))


def init(mpstate):
    return Trayectoria(mpstate)
