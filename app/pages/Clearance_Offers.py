import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "fetch_prediction_pipeline"))

import logging
import pandas as pd
import pydeck as pdk
import streamlit as st
from datetime import datetime

from store_sql import get_connection
from config import PREDICTION_THRESHOLD
from maps_utils import geocode, get_routes, nearest_stores, haversine_km

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)


# ── Cached data loading ────────────────────────────────────────────────────────

@st.cache_data(ttl=900)
def load_predictions() -> tuple[pd.DataFrame, str, str, str, str]:
    """Load pre-scored predictions from the MySQL app table. Cached for 15 minutes."""
    log.info("Cache miss — loading from MySQL app table")
    conn = get_connection()
    df   = pd.read_sql("SELECT * FROM app", conn)
    conn.close()
    log.info(f"Loaded {len(df)} scored offers from app table")

    if df.empty:
        return pd.DataFrame(), "", "", "", ""

    model_type   = df["champion_model"].iloc[0]      if "champion_model"    in df.columns else "unknown"
    trained_on   = df["champion_trained_on"].iloc[0]  if "champion_trained_on" in df.columns else "unknown"
    fetched_at   = df["fetched_at"].iloc[0]           if "fetched_at"        in df.columns else "unknown"
    predicted_at = df["predicted_at"].iloc[0]         if "predicted_at"      in df.columns else "unknown"

    df["store_lat"] = pd.to_numeric(df["store_lat"], errors="coerce")
    df["store_lng"] = pd.to_numeric(df["store_lng"], errors="coerce")
    df = df.sort_values("sell_probability", ascending=False).reset_index(drop=True)

    return df, model_type, trained_on, fetched_at, predicted_at


@st.cache_data(ttl=86400)
def geocode_cached(address: str) -> dict:
    return geocode(address)


@st.cache_data(ttl=3600)
def routes_cached(
    origin: str,
    destinations: tuple,   # tuple of (store_name, lat, lng) — hashable for cache key
    mode: str,
) -> list[dict]:
    dest_dicts = [{"store_name": n, "lat": lat, "lng": lng} for n, lat, lng in destinations]
    return get_routes(origin, dest_dicts, mode=mode)


# ── UI helpers ─────────────────────────────────────────────────────────────────

_MODE_SPEED_KMH = {"walking": 5, "bicycling": 15, "driving": 40, "transit": 25}

BRAND_LABELS = {
    "netto":          "Netto",
    "bilka":          "Bilka",
    "foetex":         "Føtex",
    "salling":        "Salling",
    "br":             "BR",
    "carlsjr":        "Carl's Jr.",
    "starbucks":      "Starbucks",
}


def _brand_label(raw: str) -> str:
    return BRAND_LABELS.get(str(raw).lower(), str(raw).title())



def _format_end_time(raw) -> str:
    try:
        dt = pd.to_datetime(raw)
        return dt.strftime("%-d %b %H:%M")
    except Exception:
        return str(raw)


def _render_card(row: pd.Series, dist_unit: str = "km") -> None:
    prob = row["sell_probability"]
    will_sell = row["will_sell"] == 1
    prob_pct = int(prob * 100)

    bar_color = "#2ecc71" if will_sell else "#e74c3c"
    label_color = "#27ae60" if will_sell else "#c0392b"
    verdict = "Will sell" if will_sell else "Won't sell"

    with st.container():
        # Product image
        img_url = row.get("product_image")
        if img_url and str(img_url) not in ("nan", "None", ""):
            st.markdown(
                f"<img src='{img_url}' style='width:100%;height:180px;"
                "object-fit:contain;background:#f0f0f0;border-radius:6px;'>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='height:180px;background:#f0f0f0;border-radius:6px;"
                "display:flex;align-items:center;justify-content:center;"
                "color:#aaa;font-size:28px;'>🛒</div>",
                unsafe_allow_html=True,
            )

        # Product name
        st.markdown(f"**{row['product_description']}**")

        cat = row.get("category_level1_en", "")
        if cat:
            st.caption(cat)

        # Store info
        brand = _brand_label(row.get("store_brand", ""))
        store = row.get("store_name", "")
        dist = row.get("distance_to_user", float("inf"))
        time = row.get("time_to_user", float("inf"))
        if dist_unit == "km" and dist != float("inf") and not pd.isna(dist):
            dist_str = f" · 📍 {dist:.1f} km"
        elif dist_unit == "min" and time != float("inf") and not pd.isna(time):
            dist_str = f" · 📍 ~{int(time)} min"
        else:
            dist_str = ""
        st.markdown(f"🏪 {brand} · {store}{dist_str}")

        # Prices + stock
        new_p  = row.get("offer_new_price", "")
        orig_p = row.get("offer_original_price", "")
        disc   = row.get("offer_percent_discount", "")
        unit   = row.get("offer_stock_unit", "")
        stock  = row.get("offer_stock", "")
        saving = row.get("offer_discount", "")

        # Format stock: kg items show 1 decimal, count items show as integer
        stock_missing = stock in ("", None) or pd.isna(stock)
        if not stock_missing and str(unit).lower() == "kg":
            try:
                stock_str = f"{float(stock):.1f} kg"
            except (ValueError, TypeError):
                stock_str = f"{stock} kg"
        elif not stock_missing:
            try:
                stock_str = f"{int(float(stock))} pcs"
            except (ValueError, TypeError):
                stock_str = str(stock)
        else:
            stock_str = ""

        saving_str = f"You save {saving} kr" if saving not in ("", None) else ""

        st.markdown(
            f"<span style='font-size:20px;font-weight:700'>{new_p} kr</span> "
            f"<span style='text-decoration:line-through;color:#888'>{orig_p} kr</span> "
            f"<span style='color:#e67e22'>-{disc}%</span><br/>"
            f"<span style='color:#27ae60;font-size:13px'>{saving_str}</span>"
            f"{'&nbsp;&nbsp;' if saving_str and stock_str else ''}"
            f"<span style='color:#999;font-size:13px'>{stock_str}</span>",
            unsafe_allow_html=True,
        )

        # Sell probability bar
        st.markdown(
            f"<div style='background:#e0e0e0;border-radius:4px;height:8px;margin:6px 0'>"
            f"<div style='background:{bar_color};width:{prob_pct}%;height:8px;border-radius:4px'></div>"
            f"</div>"
            f"<span style='color:{label_color};font-weight:600'>{verdict}</span> "
            f"<span style='color:#666'>{prob_pct}%</span>",
            unsafe_allow_html=True,
        )

        # Expiry
        end = _format_end_time(row.get("offer_end_time", ""))
        st.caption(f"Expires {end}")

        st.divider()


# ── Main app ───────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🛒 Clearance Offers - Aalborg")
    st.markdown("*Live clearance offers ranked by sell-through probability*")

    # ── Location input ─────────────────────────────────────────────────────────
    loc_col, btn_col = st.columns([5, 1])
    with loc_col:
        address_input = st.text_input(
            "📍 Your location",
            value=st.session_state.get("user_address", ""),
            placeholder="e.g. Boulevarden 13, 9000 Aalborg or Nørregade 10, 9000 Aalborg",
            label_visibility="collapsed",
        )
    with btn_col:
        locate_btn = st.button("Locate", use_container_width=True)

    _tm_opts = ["walking", "bicycling", "driving", "transit"]
    travel_mode = st.radio(
        "Travel mode",
        options=_tm_opts,
        index=_tm_opts.index(st.session_state.get("travel_mode", "walking")),
        format_func=lambda m: {"walking": "🚶 Walking", "bicycling": "🚲 Cycling",
                               "driving": "🚗 Driving", "transit": "🚌 Transit"}[m],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state["travel_mode"] = travel_mode

    if locate_btn and address_input:
        try:
            loc = geocode_cached(address_input)
            st.session_state["user_address"]  = address_input
            st.session_state["user_location"] = loc
        except Exception as exc:
            st.warning(f"Could not find that address: {exc}")
            st.session_state.pop("user_location", None)
    elif not address_input:
        st.session_state.pop("user_location", None)

    user_location = st.session_state.get("user_location")
    if user_location:
        st.caption(f"📍 {user_location['formatted_address']}")

    try:
        results, model_type, trained_on, fetched_at, predicted_at = load_predictions()
    except Exception as exc:
        if "sql-net" in str(exc) or "Can't connect" in str(exc) or "Connection" in str(exc):
            st.error("Cannot connect to database — check UCloud sql-net connection")
        elif "model.joblib" in str(exc) or "scaler.joblib" in str(exc) or "label_encoders" in str(exc):
            st.warning("No trained model found — run the ML pipeline first")
        else:
            st.error(f"Error loading predictions: {exc}")
        st.stop()

    # Header row: workflow timestamps + refresh
    col_ts, col_btn = st.columns([4, 1])
    with col_ts:
        st.caption(
            f"Fetched: {fetched_at}  ·  Predicted: {predicted_at}  ·  Cache refreshes every 15 min"
        )
    with col_btn:
        if st.button("Refresh now"):
            st.cache_data.clear()
            st.rerun()

    if results.empty:
        st.info("No clearance offers available right now. Check back later.")
        st.stop()

    # Add straight-line distance and estimated travel time to each offer's store
    speed_kmh = _MODE_SPEED_KMH.get(travel_mode, 5)
    results = results.copy()
    if user_location:
        results["distance_to_user"] = results.apply(
            lambda r: haversine_km(
                user_location["lat"], user_location["lng"],
                r["store_lat"], r["store_lng"],
            ) if pd.notna(r.get("store_lat")) and pd.notna(r.get("store_lng"))
            else float("inf"),
            axis=1,
        )
    else:
        results["distance_to_user"] = float("inf")
    results["time_to_user"] = results["distance_to_user"] / speed_kmh * 60

    # ── Sidebar filters ────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Filters")

        all_brands = sorted(results["store_brand"].dropna().unique())
        sel_brands = st.multiselect(
            "Store brand",
            options=all_brands,
            default=all_brands,
            format_func=_brand_label,
        )

        all_stores = sorted(results["store_name"].dropna().unique())
        store_dist_map = results.groupby("store_name")["distance_to_user"].first().to_dict()
        store_time_map = results.groupby("store_name")["time_to_user"].first().to_dict()

        def _store_label(name: str) -> str:
            d = store_dist_map.get(name, float("inf"))
            t = store_time_map.get(name, float("inf"))
            if d != float("inf") and not pd.isna(d):
                return f"{name}  ({d:.1f} km · ~{int(t)} min)"
            return name

        with st.expander("Store name", expanded=False):
            sel_stores = st.multiselect(
                "Stores",
                options=all_stores,
                default=all_stores,
                format_func=_store_label,
                label_visibility="collapsed",
            )

        all_cats = sorted(
            results["category_level1_en"]
            .dropna()
            .replace("", pd.NA)
            .dropna()
            .unique()
        )
        sel_cats = st.multiselect(
            "Category",
            options=all_cats,
            default=[],
            placeholder="All categories",
        )

        search_query = st.text_input(
            "Search products",
            placeholder="e.g. yoghurt, juice, ost …",
        ).strip().lower()

        verdict_filter = st.radio(
            "Show offers",
            options=["All", "Will sell", "Won't sell"],
            horizontal=True,
        )

        min_prob_pct = st.slider(
            "Min sell probability",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            format="%d%%",
        )
        min_prob = min_prob_pct / 100

        st.divider()
        sort_by = st.radio(
            "Sort by",
            options=["Sell probability", "Savings (kr)", "Savings (%)", "Closest Stores"],
            horizontal=False,
        )

        st.divider()
        st.caption("Distance filter")
        _du_opts = ["km", "min"]
        dist_unit = st.radio(
            "Distance unit",
            options=_du_opts,
            index=_du_opts.index(st.session_state.get("dist_unit", "km")),
            format_func=lambda u: "📏 Distance (km)" if u == "km" else "⏱ Travel time (min)",
            horizontal=True,
            label_visibility="collapsed",
        )
        st.session_state["dist_unit"] = dist_unit
        if dist_unit == "km":
            max_dist = st.slider(
                "Max distance", min_value=0.5, max_value=20.0,
                value=float(st.session_state.get("max_dist_km", 2.0)),
                step=0.5, format="%.1f km", label_visibility="collapsed",
            )
            st.session_state["max_dist_km"] = max_dist
        else:
            max_dist = st.slider(
                "Max travel time", min_value=5, max_value=120,
                value=int(st.session_state.get("max_dist_min", 15)),
                step=5, format="%d min", label_visibility="collapsed",
            )
            st.session_state["max_dist_min"] = max_dist
        if user_location:
            nearby_label = (
                f"Only offers within {max_dist:.1f} km"
                if dist_unit == "km"
                else f"Only offers within ~{int(max_dist)} min"
            )
            only_nearby = st.checkbox(nearby_label, value=True)
        else:
            only_nearby = False
            st.caption("Set your location to enable distance filtering")

    # Apply filters
    df = results.copy()
    if sel_brands:
        df = df[df["store_brand"].isin(sel_brands)]
    if sel_stores:
        df = df[df["store_name"].isin(sel_stores)]
    if sel_cats:
        df = df[df["category_level1_en"].isin(sel_cats)]
    if search_query:
        name_match = df["product_description"].fillna("").str.lower().str.contains(search_query, regex=False)
        cat_match  = df["category_level1_en"].fillna("").str.lower().str.contains(search_query, regex=False)
        df = df[name_match | cat_match]
    if verdict_filter == "Will sell":
        df = df[df["will_sell"] == 1]
    elif verdict_filter == "Won't sell":
        df = df[df["will_sell"] == 0]
    df = df[df["sell_probability"] >= min_prob]
    if only_nearby and user_location:
        filter_col = "distance_to_user" if dist_unit == "km" else "time_to_user"
        df = df[df[filter_col] <= max_dist]

    # Apply sort
    if sort_by == "Savings (kr)":
        df = df.sort_values("offer_discount", ascending=False).reset_index(drop=True)
    elif sort_by == "Savings (%)":
        df = df.sort_values("offer_percent_discount", ascending=False).reset_index(drop=True)
    elif sort_by == "Closest Stores" and user_location:
        sort_col = "distance_to_user" if dist_unit == "km" else "time_to_user"
        df = df.sort_values(sort_col, ascending=True).reset_index(drop=True)
    else:
        df = df.sort_values("sell_probability", ascending=False).reset_index(drop=True)

    # ── Summary metrics ────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total offers", len(df))
    m2.metric("Will sell", int(df["will_sell"].sum()))
    m3.metric("Won't sell", int((df["will_sell"] == 0).sum()))
    m4.metric("Avg probability", f"{df['sell_probability'].mean():.0%}" if len(df) else "—")

    st.divider()

    # ── Store map ──────────────────────────────────────────────────────────────
    with st.expander("Store map — hover a pin to see offers", expanded=True):
        # Use full unfiltered results for store pins so filters don't hide stores
        geo = results.dropna(subset=["store_lat", "store_lng"])
        if geo.empty:
            st.caption("No location data available.")
        else:
            store_rows = []
            for (lat, lng, name, brand), grp in geo.groupby(
                ["store_lat", "store_lng", "store_name", "store_brand"], sort=False
            ):
                n_total  = len(grp)
                n_sell   = int(grp["will_sell"].sum())
                sell_pct = n_sell / n_total if n_total else 0

                if sell_pct >= 0.5:
                    color = [46, 204, 113, 230]
                elif sell_pct >= 0.2:
                    color = [230, 126, 34, 230]
                else:
                    color = [231, 76, 60, 230]

                lines = []
                for _, r in grp.sort_values("sell_probability", ascending=False).head(6).iterrows():
                    verdict   = "✓" if r["will_sell"] == 1 else "✗"
                    sell_prob = int(r["sell_probability"] * 100)
                    disc      = r.get("offer_percent_discount", "")
                    price     = r.get("offer_new_price", "")
                    lines.append(
                        f"{verdict} {r['product_description']}<br/>"
                        f"&nbsp;&nbsp;&nbsp;"
                        f"<span style='color:#888'>{price} kr · -{disc}% off</span> · "
                        f"<span style='color:#3498db'>sell prob: {sell_prob}%</span>"
                    )
                offers_html = "<br/>".join(lines)
                if n_total > 6:
                    offers_html += f"<br/><i>… and {n_total - 6} more</i>"

                tooltip_html = (
                    f"<b style='font-size:14px'>{name}</b> "
                    f"<span style='color:#888'>({_brand_label(brand)})</span><br/>"
                    f"<b>{n_total}</b> offers &nbsp;·&nbsp; "
                    f"<b style='color:#27ae60'>{n_sell} predicted to sell</b>"
                    f"<hr style='margin:6px 0;border-color:#ddd'/>"
                    f"{offers_html}"
                )
                store_rows.append({
                    "lat": lat, "lon": lng,
                    "store_name": name, "store_brand": _brand_label(brand),
                    "n_total": n_total, "n_sell": n_sell,
                    "color": color, "tooltip_html": tooltip_html,
                })

            store_map_df = pd.DataFrame(store_rows)
            layers = [
                pdk.Layer(
                    "ScatterplotLayer",
                    data=store_map_df,
                    get_position="[lon, lat]",
                    get_fill_color="color",
                    get_radius=80,
                    radius_min_pixels=8,
                    radius_max_pixels=24,
                    pickable=True,
                )
            ]

            center_lat = store_map_df["lat"].mean()
            center_lng = store_map_df["lon"].mean()

            # ── Route overlay when user location is set ────────────────────────
            routes = []
            if user_location:
                near = nearest_stores(
                    user_location["lat"], user_location["lng"],
                    [{"store_name": r["store_name"], "lat": r["lat"], "lng": r["lon"]}
                     for r in store_rows],
                    n=len(store_rows),
                )
                # Generous haversine pre-filter to limit API calls (road distance is always longer)
                haversine_limit_km = (
                    max_dist * 1.5
                    if dist_unit == "km"
                    else (max_dist / 60) * speed_kmh * 1.5
                )
                near = [s for s in near if s["straight_km"] <= haversine_limit_km]
                if near:
                    dest_key = tuple((s["store_name"], s["lat"], s["lng"]) for s in near)
                    try:
                        routes = routes_cached(
                            user_location["formatted_address"], dest_key, travel_mode
                        )
                    except Exception:
                        routes = []

                # Precise post-filter using actual route data from the API
                if routes:
                    if dist_unit == "km":
                        routes = [
                            r for r in routes
                            if r.get("distance_meters") and r["distance_meters"] / 1000 <= max_dist
                        ]
                    else:
                        routes = [
                            r for r in routes
                            if r.get("duration_seconds") and r["duration_seconds"] / 60 <= max_dist
                        ]

                if routes:
                    mode_label = {"walking": "🚶 Walking", "bicycling": "🚲 Cycling",
                                  "driving": "🚗 Driving", "transit": "🚌 Transit"}.get(travel_mode, travel_mode)
                    route_color = [52, 152, 219, 220]
                    path_data = [
                        {
                            "path":  r["polyline"],
                            "color": route_color,
                            "tooltip_html": (
                                f"<b>Distance to {r['store_name']}</b><br/>"
                                f"{r['distance_text']} &nbsp;·&nbsp; {r['duration_text']}<br/>"
                                f"<span style='color:#888'>{mode_label}</span>"
                            ),
                        }
                        for r in routes if r["polyline"]
                    ]
                    layers.append(
                        pdk.Layer(
                            "PathLayer",
                            data=path_data,
                            get_path="path",
                            get_color="color",
                            get_width=4,
                            width_min_pixels=3,
                            pickable=True,
                        )
                    )

                # User location pin — not pickable so it doesn't trigger the store tooltip
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=[{"lat": user_location["lat"], "lon": user_location["lng"]}],
                        get_position="[lon, lat]",
                        get_fill_color=[52, 152, 219, 255],
                        get_line_color=[255, 255, 255, 255],
                        get_radius=100,
                        radius_min_pixels=10,
                        radius_max_pixels=20,
                        stroked=True,
                        line_width_min_pixels=3,
                        pickable=False,
                    )
                )
                center_lat = user_location["lat"]
                center_lng = user_location["lng"]

            tooltip = {
                "html": "{tooltip_html}",
                "style": {
                    "backgroundColor": "white",
                    "color": "#333",
                    "padding": "12px",
                    "borderRadius": "8px",
                    "fontSize": "13px",
                    "maxWidth": "320px",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.15)",
                },
            }

            st.pydeck_chart(
                pdk.Deck(
                    layers=layers,
                    initial_view_state=pdk.ViewState(
                        latitude=center_lat,
                        longitude=center_lng,
                        zoom=12,
                        pitch=0,
                    ),
                    tooltip=tooltip,
                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                ),
                use_container_width=True,
            )
            legend = "🟢 ≥50% sell  🟠 20–50%  🔴 <20%"
            if user_location:
                legend += "  ·  🔵 Your location"
            st.caption(legend)

            # ── Nearest store summary ──────────────────────────────────────────
            if routes:
                SUMMARY_COLS = 4
                for row_start in range(0, len(routes), SUMMARY_COLS):
                    chunk = routes[row_start : row_start + SUMMARY_COLS]
                    cols_r = st.columns(SUMMARY_COLS)
                    for col_r, route in zip(cols_r, chunk):
                        with col_r:
                            st.markdown(
                                f"<div style='border:1px solid #333;border-radius:8px;"
                                f"padding:10px 14px;margin-bottom:4px'>"
                                f"<div style='font-weight:600;font-size:14px'>{route['store_name']}</div>"
                                f"<div style='color:#888;font-size:12px;margin-top:4px'>"
                                f"🗺 {route['distance_text']} &nbsp;·&nbsp; ⏱ {route['duration_text']}"
                                f"</div></div>",
                                unsafe_allow_html=True,
                            )

    st.divider()

    # ── Offer cards ────────────────────────────────────────────────────────────
    if df.empty:
        st.info("No offers match your filters.")
        st.stop()

    COLS = 3
    rows = [df.iloc[i : i + COLS] for i in range(0, len(df), COLS)]
    for chunk in rows:
        cols = st.columns(COLS)
        for col, (_, row) in zip(cols, chunk.iterrows()):
            with col:
                _render_card(row, dist_unit=dist_unit)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.caption(f"Model: {model_type}  ·  Trained: {trained_on}  ·  Threshold: {PREDICTION_THRESHOLD}")


if __name__ == "__main__":
    main()
