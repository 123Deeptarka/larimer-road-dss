# -*- coding: utf-8 -*-
"""
Street-level imagery for the DSS — Google Street View Static API.

OpenStreetMap carries no photography, so imagery comes from Street View. It has
near-complete coverage on public roads, but requires a billing-enabled key, and
the Maps Platform terms restrict automated processing and storage of the
imagery — check those terms before publishing derived results.

NOTE ON VALIDITY: Model_5 was benchmarked on Xu et al.'s road-surface
photographs — close views where pavement fills the frame. Street View is
wide-angle and oblique, with pavement occupying a smaller, foreshortened part of
the image. PSCI grades from this imagery are outside the benchmark's domain, and
its reported MAE does not automatically transfer. Pitching the camera down
(DEFAULT_PITCH) mitigates this but does not eliminate it.
"""

import requests

SVS_IMAGE = "https://maps.googleapis.com/maps/api/streetview"
SVS_META = "https://maps.googleapis.com/maps/api/streetview/metadata"

DEFAULT_RADIUS_M = 60
DEFAULT_PITCH = -40           # look down at the pavement, not the horizon
DEFAULT_FOV = 90
DEFAULT_SIZE = "640x640"
TIMEOUT = 30


def streetview_metadata(lat, lon, key, radius_m=DEFAULT_RADIUS_M):
    """Check coverage before requesting an image. This endpoint is free —
    always call it first so unbilled ZERO_RESULTS cost nothing."""
    r = requests.get(SVS_META, params={"location": f"{lat},{lon}",
                                       "radius": int(radius_m), "key": key},
                     timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def streetview_fetch(lat, lon, key, heading=None, pitch=DEFAULT_PITCH,
                     fov=DEFAULT_FOV, size=DEFAULT_SIZE,
                     radius_m=DEFAULT_RADIUS_M):
    """Return (image_bytes, media_type, metadata), or (None, None, metadata)
    when no panorama exists within `radius_m`."""
    meta = streetview_metadata(lat, lon, key, radius_m)
    if meta.get("status") != "OK":
        return None, None, meta

    params = {"size": size, "location": f"{lat},{lon}", "pitch": pitch,
              "fov": fov, "radius": int(radius_m), "return_error_code": "true",
              "key": key}
    if heading is not None:
        params["heading"] = round(float(heading), 1)

    r = requests.get(SVS_IMAGE, params=params, timeout=TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"Street View image failed ({r.status_code}): {r.text[:200]}")
    return r.content, r.headers.get("Content-Type", "image/jpeg"), meta
