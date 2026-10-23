"""Pruebas de la conversión del marco del mundo (ENU) a WGS84."""
import math

import pytest

from cuadrirotor_sim.geo import ORIGEN, enu_a_geodesica, texto_dms


def test_origen():
    assert enu_a_geodesica(0, 0, 0) == pytest.approx(ORIGEN, abs=1e-6)


def test_distancias():
    lat0, lon0, _ = ORIGEN
    lat, lon, h = enu_a_geodesica(0.0, 1000.0, 0.0)      # 1 km al norte
    assert (lat - lat0) * 110_780 == pytest.approx(1000.0, rel=2e-3)   # m por grado a 19°
    assert lon == pytest.approx(lon0, abs=1e-9)
    lat, lon, h = enu_a_geodesica(1000.0, 0.0, 0.0)      # 1 km al este
    m_por_grado = math.pi / 180 * 6378137.0 * math.cos(math.radians(lat0))
    assert (lon - lon0) * m_por_grado == pytest.approx(1000.0, rel=2e-3)
    assert h == pytest.approx(ORIGEN[2] + 0.08, abs=0.02)   # curvatura: 1000²/(2R)


def test_altura():
    assert enu_a_geodesica(0, 0, 25.0)[2] == pytest.approx(ORIGEN[2] + 25.0, abs=1e-6)


def test_dms():
    assert texto_dms(19.5, -99.25) == '19°30\'00.0"N 99°15\'00.0"W'
