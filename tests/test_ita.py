import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ita import ita_value, ita_group


def fake(rgb):
    return np.tile(np.array(rgb, dtype=np.uint8), (64, 64, 1))


def test_light_skin_has_higher_ita_than_dark():
    assert ita_value(fake((235, 200, 180))) > ita_value(fake((120, 80, 60))) > ita_value(fake((60, 35, 25)))


def test_groups():
    assert ita_group(60) == "lighter" and ita_group(20) == "medium" and ita_group(-40) == "darker"
