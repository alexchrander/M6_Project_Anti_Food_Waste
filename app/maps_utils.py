"""
maps_utils.py — Google Maps API helpers for geocoding and routes.
Place in app/ alongside app.py.

APIs used:
  - Geocoding API     : address → lat/lng
  - Routes API        : single request for all routes + distances (replaces
                        Directions API and Distance Matrix API)
"""

import math
import os
import requests
import polyline
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GEOCODING_URL = "https://maps.googleapis.com/maps/api/geocode/json"
ROUTES_URL    = "https://routes.googleapis.com/directions/v2:computeRoutes"

TRAVEL_MODES = ["walking", "bicycling", "transit", "driving"]

# Routes API uses different mode names than Directions API
_MODE_MAP = {
    "walking":   "WALK",
    "bicycling": "BICYCLE",
    "transit":   "TRANSIT",
    "driving":   "DRIVE",
}


def get_api_key() -> str:
    key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    if not key:
        raise ValueError("GOOGLE_MAPS_API_KEY not found in environment / .env")
    return key


def geocode(address: str) -> dict:
    """Convert a human-readable address to lat/lng + formatted address."""
    resp = requests.get(
        GEOCODING_URL,
        params={"address": address, "key": get_api_key()},
    )
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "OK":
        raise ValueError(f"Geocoding failed: {data['status']} — {data.get('error_message', '')}")
    result   = data["results"][0]
    location = result["geometry"]["location"]
    return {
        "formatted_address": result["formatted_address"],
        "lat": location["lat"],
        "lng": location["lng"],
    }


def get_routes(
    origin_address: str,
    destinations: list[dict],
    mode: str = "walking",
) -> list[dict]:
    """
    Fetch road-following routes from one origin to multiple destinations
    in a single Routes API request.

    Parameters
    ----------
    origin_address : str
        Formatted address string of the user's location.
    destinations : list of dict
        Each dict must have keys: 'store_name', 'lat', 'lng', and any
        other fields you want passed through (e.g. 'distance_text').
    mode : str
        One of TRAVEL_MODES.

    Returns
    -------
    list of dict, one per destination, each containing:
        store_name, lat, lng, distance_meters, distance_text,
        duration_seconds, duration_text, polyline ([lng, lat] pairs)
    """
    api_key     = get_api_key()
    travel_mode = _MODE_MAP.get(mode, "WALK")

    results = []
    for dest in destinations:
        body = {
            "origin": {
                "address": origin_address,
            },
            "destination": {
                "location": {
                    "latLng": {
                        "latitude":  dest["lat"],
                        "longitude": dest["lng"],
                    }
                }
            },
            "travelMode": travel_mode,
            "computeAlternativeRoutes": False,
            "routeModifiers": {
                "avoidTolls": False,
                "avoidHighways": False,
                "avoidFerries": False,
            },
        }

        headers = {
            "Content-Type":         "application/json",
            "X-Goog-Api-Key":       api_key,
            # Request only the fields we need to minimise response size
            "X-Goog-FieldMask": (
                "routes.distanceMeters,"
                "routes.duration,"
                "routes.polyline.encodedPolyline"
            ),
        }

        resp = requests.post(ROUTES_URL, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        routes = data.get("routes", [])
        if not routes:
            results.append({
                **dest,
                "distance_meters":  None,
                "distance_text":    "N/A",
                "duration_seconds": None,
                "duration_text":    "N/A",
                "polyline":         [],
            })
            continue

        route            = routes[0]
        distance_m       = route.get("distanceMeters", 0)
        duration_s_str   = route.get("duration", "0s").rstrip("s")
        duration_s       = int(float(duration_s_str))
        encoded          = route["polyline"]["encodedPolyline"]
        coords           = [[lng, lat] for lat, lng in polyline.decode(encoded)]

        # Human-readable formatting
        distance_km   = distance_m / 1000
        distance_text = f"{distance_km:.1f} km" if distance_km >= 1 else f"{distance_m} m"
        duration_min  = duration_s // 60
        duration_text = f"{duration_min} min" if duration_min < 60 else f"{duration_min // 60}h {duration_min % 60}min"

        results.append({
            **dest,
            "distance_meters":  distance_m,
            "distance_text":    distance_text,
            "duration_seconds": duration_s,
            "duration_text":    duration_text,
            "polyline":         coords,
        })

    return results


# ── Distance helpers ──────────────────────────────────────────────────────────

def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Straight-line distance in km between two lat/lng points."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlng / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def nearest_stores(
    user_lat: float,
    user_lng: float,
    stores: list[dict],
    n: int = 3,
) -> list[dict]:
    """Return the n closest stores by straight-line distance.

    Each store dict must have 'lat' and 'lng' keys.
    Returns dicts with 'straight_km' added.
    """
    ranked = sorted(
        stores,
        key=lambda s: haversine_km(user_lat, user_lng, s["lat"], s["lng"]),
    )
    return [
        {**s, "straight_km": haversine_km(user_lat, user_lng, s["lat"], s["lng"])}
        for s in ranked[:n]
    ]