# -*- coding: utf-8 -*-
"""
Pure scoring and image helpers shared by the DSS app and the agent.

Kept separate from dss_app.py deliberately: that module executes Streamlit
calls at import time, so importing it from the agent would run the whole UI
(and create a circular import once the app calls the agent). Everything here
is side-effect free.
"""

import base64
import io
import os
import re

PSCI_MIN, PSCI_MAX = 1, 10
UNPARSEABLE = 11                   # matches Model_5's sentinel for a bad reply

# Status palette (fixed, never themed). Contrast measured on Streamlit's own
# surfaces: warning 1.83 and serious 2.64 fall below 3:1 on white, so every tier
# ships with an ICON + LABEL and colour never carries the meaning alone.
TIERS = [
    (0.75, "Critical", "#d03b3b", "▲"),
    (0.50, "High",     "#ec835a", "◆"),
    (0.25, "Medium",   "#fab219", "●"),
    (0.00, "Low",      "#0ca30c", "▪"),
]

MAX_IMAGE_PX = 1600
JPEG_QUALITY = 85


def psci_to_urgency(psci):
    """PSCI 1-10 (10 = best) -> condition urgency in [0,1] (1.0 = failed)."""
    return 1.0 - (psci - PSCI_MIN) / (PSCI_MAX - PSCI_MIN)


def net_score(urgency, importance, w_cond, w_imp):
    """Weighted fusion, renormalized so the result stays in [0,1] even if the
    weights do not sum to 1."""
    return (w_cond * urgency + w_imp * importance) / (w_cond + w_imp)


def classify(score):
    for cutoff, label, color, icon in TIERS:
        if score >= cutoff:
            return label, color, icon
    return TIERS[-1][1], TIERS[-1][2], TIERS[-1][3]


def extract_grade(text):
    """First integer 1-10 in the reply; UNPARSEABLE if none. Identical to
    Model_5's parser so app and benchmark agree."""
    m = re.search(r"\b(10|11|[1-9])\b", text)
    return int(m.group(1)) if m else UNPARSEABLE


def prepare_image(raw, name="image.jpg"):
    """Downscale to MAX_IMAGE_PX and re-encode as JPEG -> (b64, media_type)."""
    try:
        from PIL import Image
    except ImportError:
        ext = os.path.splitext(name)[1].lower()
        media = {".png": "image/png", ".gif": "image/gif",
                 ".webp": "image/webp"}.get(ext, "image/jpeg")
        return base64.standard_b64encode(raw).decode("utf-8"), media

    img = Image.open(io.BytesIO(raw))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if max(img.size) > MAX_IMAGE_PX:
        scale = MAX_IMAGE_PX / max(img.size)
        img = img.resize((max(1, int(img.width * scale)),
                          max(1, int(img.height * scale))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"
