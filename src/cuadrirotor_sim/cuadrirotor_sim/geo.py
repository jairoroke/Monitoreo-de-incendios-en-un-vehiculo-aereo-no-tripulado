"""Conversión del marco local del mundo (ENU: x este, y norte, z arriba) a WGS84.

Gazebo usa la misma convención con <spherical_coordinates> y
<world_frame_orientation>ENU</world_frame_orientation>, así que un sensor
NavSat del dron daría las mismas coordenadas que estas funciones.
"""
import math

# Origen del mundo: Parque Nacional Desierto de los Leones (CDMX), un bosque de
# oyamel con incendios casi cada temporada seca. Es un lugar supuesto para la
# simulación: el valle del mundo 'bosque_grande' no reproduce su relieve.
ORIGEN = (19.2930, -99.3110, 3000.0)   # latitud [°], longitud [°], altitud [m]

A = 6378137.0                 # semieje mayor WGS84 [m]
E2 = 6.69437999014e-3         # excentricidad al cuadrado


def _ecef(lat, lon, h):
    la, lo = math.radians(lat), math.radians(lon)
    n = A / math.sqrt(1 - E2 * math.sin(la) ** 2)
    return ((n + h) * math.cos(la) * math.cos(lo),
            (n + h) * math.cos(la) * math.sin(lo),
            (n * (1 - E2) + h) * math.sin(la))


def enu_a_geodesica(x, y, z, origen=ORIGEN):
    """(latitud °, longitud °, altitud m) del punto (x, y, z) del mundo."""
    lat0, lon0, h0 = origen
    la, lo = math.radians(lat0), math.radians(lon0)
    x0, y0, z0 = _ecef(lat0, lon0, h0)
    sl, cl, so, co = math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)
    X = x0 - so * x - sl * co * y + cl * co * z
    Y = y0 + co * x - sl * so * y + cl * so * z
    Z = z0 + cl * y + sl * z
    # ECEF -> geodésicas (iteración de Bowring; converge en 2 o 3 pasos).
    lon = math.atan2(Y, X)
    p = math.hypot(X, Y)
    lat = math.atan2(Z, p * (1 - E2))
    for _ in range(5):
        n = A / math.sqrt(1 - E2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - n
        lat = math.atan2(Z, p * (1 - E2 * n / (n + h)))
    n = A / math.sqrt(1 - E2 * math.sin(lat) ** 2)
    h = p / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), h


def texto_dms(lat, lon):
    """19°17'34.8"N 99°18'39.6"W"""
    def dms(v, pos, neg):
        s = pos if v >= 0 else neg
        v = abs(v)
        g = int(v)
        m = int((v - g) * 60)
        return f"{g}°{m:02d}'{(v - g - m / 60) * 3600:04.1f}\"{s}"
    return f'{dms(lat, "N", "S")} {dms(lon, "E", "W")}'
