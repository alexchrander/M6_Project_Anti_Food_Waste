"""
test_maps_app.py — Minimal Streamlit app to test Google Maps Routes API integration.

Run with:
    streamlit run app/test_maps_app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
import pydeck as pdk
import streamlit as st

from maps_utils import geocode, get_routes, TRAVEL_MODES

st.set_page_config(page_title="Maps API Test", page_icon="🗺️", layout="wide")

# ── Real store data ────────────────────────────────────────────────────────────

STORES = [
    {"store_name": "Netto Aalborg",             "store_brand": "netto",  "lat": 57.0230, "lng": 9.87497},
    {"store_name": "Netto Danmarksgd. Aalborg", "store_brand": "netto",  "lat": 57.0453, "lng": 9.92405},
    {"store_name": "Netto Eternitten",          "store_brand": "netto",  "lat": 57.0363, "lng": 9.93591},
    {"store_name": "Netto Hadsundvej Aalborg",  "store_brand": "netto",  "lat": 57.0374, "lng": 9.95352},
    {"store_name": "Netto Kollegievej",         "store_brand": "netto",  "lat": 57.0247, "lng": 9.94331},
    {"store_name": "Netto Vesterbro Aalborg",   "store_brand": "netto",  "lat": 57.0506, "lng": 9.91621},
    {"store_name": "Netto Vestre Allé",         "store_brand": "netto",  "lat": 57.0331, "lng": 9.90752},
    {"store_name": "føtex Aalborg",             "store_brand": "foetex", "lat": 57.0473, "lng": 9.92423},
    {"store_name": "føtex Eternitten",          "store_brand": "foetex", "lat": 57.0375, "lng": 9.93532},
    {"store_name": "føtex Food Hasseris",       "store_brand": "foetex", "lat": 57.0359, "lng": 9.88336},
]

BRAND_COLORS = {
    "netto":  [255, 220, 0, 220],
    "foetex": [0, 100, 200, 220],
}
USER_COLOR   = [231, 76, 60, 255]
ROUTE_COLOR  = [46, 204, 113, 220]
ROUTE_COLORS = [ROUTE_COLOR] * 3

# ── UI ─────────────────────────────────────────────────────────────────────────

st.title("🗺️ Maps API — Integration Test")
st.markdown("Enter your address to find the 3 closest stores and see road routes.")

with st.form("maps_form"):
    address_input = st.text_input(
        "Your address",
        placeholder="e.g. Boulevarden 13, 9000 Aalborg",
    )
    mode = st.radio(
        "Travel mode",
        options=TRAVEL_MODES,
        horizontal=True,
        format_func=lambda m: {
            "walking":   "🚶 Walking",
            "bicycling": "🚲 Bicycling",
            "transit":   "🚌 Transit",
            "driving":   "🚗 Driving",
        }[m],
    )
    submitted = st.form_submit_button("Find closest stores", use_container_width=True)

if not submitted:
    st.stop()

if not address_input.strip():
    st.error("Please enter an address.")
    st.stop()

# ── Step 1: Geocode ────────────────────────────────────────────────────────────

with st.spinner("Geocoding your address..."):
    try:
        user_geo = geocode(address_input.strip())
    except Exception as e:
        st.error(f"Geocoding failed: {e}")
        st.stop()

st.success(f"📍 **{user_geo['formatted_address']}** ({user_geo['lat']:.5f}, {user_geo['lng']:.5f})")

# ── Step 2: Get all routes in one request ──────────────────────────────────────

with st.spinner(f"Fetching {mode} routes to all stores..."):
    try:
        all_routes = get_routes(user_geo["formatted_address"], STORES, mode=mode)
    except Exception as e:
        st.error(f"Routes API failed: {e}")
        st.stop()

# Sort by distance, take 3 closest
all_routes.sort(key=lambda x: x["distance_meters"] or float("inf"))
closest = all_routes[:3]

# ── Step 3: Summary metrics ────────────────────────────────────────────────────

st.subheader(f"3 closest stores · {mode}")
cols = st.columns(3)
for col, store in zip(cols, closest):
    with col:
        st.markdown(
            f"<div style='border-left: 4px solid #2ecc71; padding-left: 10px'>"
            f"<b>{store['store_name']}</b><br/>"
            f"<span style='font-size:22px;font-weight:700'>{store['distance_text']}</span>"
            f"&nbsp;&nbsp;<span style='color:#888'>{store['duration_text']}</span><br/>"
            f"<span style='color:#888;font-size:13px'>{store['store_brand']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ── Step 4: Map ────────────────────────────────────────────────────────────────

st.subheader("Map")

# User dot
user_df = pd.DataFrame([{
    "lat":   user_geo["lat"],
    "lng":   user_geo["lng"],
    "label": "📍 You",
    "info":  user_geo["formatted_address"],
    "color": USER_COLOR,
    "radius": 60,
}])

# All store dots
closest_names = {s["store_name"] for s in closest}
store_df = pd.DataFrame([{
    "lat":    s["lat"],
    "lng":    s["lng"],
    "label":  s["store_name"],
    "info":   s["store_brand"],
    "color":  BRAND_COLORS.get(s["store_brand"], [100, 100, 100, 200]),
    "radius": 70 if s["store_name"] in closest_names else 45,
} for s in STORES])

all_points_df = pd.concat([user_df, store_df], ignore_index=True)

# Route path layer — one row per closest store
routes_df = pd.DataFrame([{
    "path":  s["polyline"],
    "store": s["store_name"],
    "info":  f"{s['distance_text']} · {s['duration_text']}",
    "color": ROUTE_COLORS[i],
} for i, s in enumerate(closest) if s["polyline"]])

path_layer = pdk.Layer(
    "PathLayer",
    data=routes_df,
    get_path="path",
    get_color="color",
    get_width=5,
    width_min_pixels=3,
    pickable=True,
)

scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=all_points_df,
    get_position="[lng, lat]",
    get_fill_color="color",
    get_radius="radius",
    radius_min_pixels=8,
    radius_max_pixels=22,
    pickable=True,
)

all_lats = [user_geo["lat"]] + [s["lat"] for s in STORES]
all_lngs = [user_geo["lng"]] + [s["lng"] for s in STORES]

view = pdk.ViewState(
    latitude=sum(all_lats) / len(all_lats),
    longitude=sum(all_lngs) / len(all_lngs),
    zoom=12,
    pitch=0,
)

tooltip = {
    "html": "<b>{label}</b><br/><span style='color:#888'>{info}</span>",
    "style": {
        "backgroundColor": "white",
        "color": "#333",
        "padding": "8px",
        "borderRadius": "6px",
        "fontSize": "13px",
    },
}

st.pydeck_chart(
    pdk.Deck(
        layers=[path_layer, scatter_layer],
        initial_view_state=view,
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    ),
    use_container_width=True,
)

st.caption("🔴 You  · 🟡 Netto  · 🔵 føtex  · Coloured lines = 3 closest routes")