"""Individual Typology Angle (ITA) skin-tone estimate.

ITA = arctan((L* - 50) / b*) * 180 / pi, with L* (lightness) and b* (yellow-blue) from CIELAB.
Higher ITA = lighter skin. Common literature bands (Del Bino & Bernerd 2013):
  very light > 55, light 41-55, intermediate 28-41, tan 10-28, brown -30-10, dark < -30.
We map them to three groups: lighter (very light+light), medium (intermediate+tan), darker (brown+dark).
Cut-offs are CHECKED on Fitzpatrick17k and may be changed there only (never on ISIC test images).

Skin pixels: dermoscopy images have a black circular border and a lesion in the middle, so we use a ring
of pixels between 55% and 90% of the half-width from the centre and drop very dark pixels (border).
"""
import numpy as np
from skimage.color import rgb2lab

DEFAULT_CUTS = (28.0, 41.0)          # ITA > 41 lighter; 28..41 medium(upper); <28 -> see below


def skin_pixels(rgb, inner=0.55, outer=0.90, min_l=15):
    h, w, _ = rgb.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((yy - h / 2) / (h / 2), (xx - w / 2) / (w / 2))
    mask = (r >= inner) & (r <= outer)
    lab = rgb2lab(rgb[mask][None, :, :] / 255.0)[0]
    return lab[lab[:, 0] > min_l]


def ita_value(rgb):
    lab = skin_pixels(rgb)
    if len(lab) < 50:
        return float("nan")
    L, b = np.median(lab[:, 0]), np.median(lab[:, 2])
    return float(np.degrees(np.arctan2(L - 50.0, b)))


def ita_group(ita, cuts=(10.0, 41.0)):
    """lighter if ITA > cuts[1]; darker if ITA <= cuts[0]; else medium. (10 and 41 = literature band edges.)"""
    if np.isnan(ita):
        return "unknown"
    return "lighter" if ita > cuts[1] else ("darker" if ita <= cuts[0] else "medium")
