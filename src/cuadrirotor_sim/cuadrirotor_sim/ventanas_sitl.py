"""Acomoda las ventanas de la simulación con ArduPilot SITL y deja al frente la
terminal de MAVProxy (donde se escriben los comandos: arm throttle, mode ...).

El gestor de ventanas abre Gazebo, el mapa y la consola de MAVProxy encima de
las terminales. Este script espera a que aparezcan (hasta 'espera' s), las
coloca en columnas según el tamaño de la pantalla y activa la de MAVProxy:

  izquierda   terminal MAVProxy (arriba), consola de MAVProxy, terminal ArduCopter
  centro      Gazebo
  derecha     mapa de MAVProxy

Usa EWMH (_NET_MOVERESIZE_WINDOW, _NET_ACTIVE_WINDOW), que respetan GNOME y la
mayoría de los gestores. Necesita python-xlib (está en ~/venv-ardupilot).

  python3 -m cuadrirotor_sim.ventanas_sitl [espera]
"""
import sys
import time

from Xlib import X, display, protocol

# Prefijo del título -> nombre de la casilla.
TITULOS = {'MAVProxy': 'terminal', 'Console': 'consola', 'ArduCopter': 'ardupilot',
           'Gazebo Sim': 'gazebo', 'Map': 'mapa'}


def ventanas(d):
    raiz = d.screen().root
    lista = raiz.get_full_property(d.intern_atom('_NET_CLIENT_LIST'), X.AnyPropertyType)
    nombre = d.intern_atom('_NET_WM_NAME')
    utf8 = d.intern_atom('UTF8_STRING')
    halladas = {}
    for wid in (lista.value if lista else []):
        w = d.create_resource_object('window', wid)
        try:
            p = w.get_full_property(nombre, utf8)
            titulo = p.value.decode('utf-8', 'replace') if p else (w.get_wm_name() or '')
        except Exception:
            continue
        for prefijo, casilla in TITULOS.items():
            if titulo.startswith(prefijo) and casilla not in halladas:
                halladas[casilla] = w
    return halladas


def mensaje(d, w, tipo, datos):
    raiz = d.screen().root
    ev = protocol.event.ClientMessage(window=w, client_type=d.intern_atom(tipo),
                                      data=(32, (datos + [0] * 5)[:5]))
    raiz.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)


def mover(d, w, x, y, ancho, alto):
    # Quitar la maximización, si la tiene, para poder mover la ventana.
    maxv, maxh = d.intern_atom('_NET_WM_STATE_MAXIMIZED_VERT'), d.intern_atom('_NET_WM_STATE_MAXIMIZED_HORZ')
    mensaje(d, w, '_NET_WM_STATE', [0, maxv, maxh, 1])
    # Banderas: posición y tamaño dados, fuente = aplicación normal (1 << 12).
    mensaje(d, w, '_NET_MOVERESIZE_WINDOW', [(1 << 12) | (0xF << 8) | 1, x, y, ancho, alto])


def main():
    espera = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
    d = display.Display()
    pantalla = d.screen()
    W, H = pantalla.width_in_pixels, pantalla.height_in_pixels
    t0 = time.monotonic()
    halladas = {}
    while time.monotonic() - t0 < espera:
        halladas = ventanas(d)
        if all(k in halladas for k in ('terminal', 'consola', 'mapa', 'gazebo', 'ardupilot')):
            break
        time.sleep(1.0)
    time.sleep(2.0)    # que terminen de abrirse
    halladas = ventanas(d)

    izq = min(1000, W // 3)
    der = min(900, W // 4) if W >= 2400 else 0
    centro = W - izq - der
    alto_term = int(H * 0.45)
    alto_cons = int(H * 0.25)
    casillas = {
        'terminal': (0, 0, izq, alto_term),
        'consola': (0, alto_term, izq, alto_cons),
        'ardupilot': (0, alto_term + alto_cons, izq, H - alto_term - alto_cons),
        'gazebo': (izq, 0, centro, H),
        'mapa': (izq + centro, 0, der, int(H * 0.7)) if der else (izq, int(H * 0.5), centro, H // 2),
    }
    for casilla, w in halladas.items():
        mover(d, w, *casillas[casilla])
    d.flush()
    if 'terminal' in halladas:
        time.sleep(0.5)
        mensaje(d, halladas['terminal'], '_NET_ACTIVE_WINDOW', [2, X.CurrentTime, 0])
        d.flush()
    faltan = [k for k in casillas if k not in halladas]
    print(f'Ventanas acomodadas en {W}x{H}' + (f'; no aparecieron: {", ".join(faltan)}' if faltan else ''))


if __name__ == '__main__':
    main()
