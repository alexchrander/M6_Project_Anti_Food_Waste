import sys
from pathlib import Path

# Add project root and fetch_pipeline to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "fetch_pipeline"))

import logging
import pandas as pd
import pydeck as pdk
import streamlit as st
from datetime import datetime

from store_sql import get_connection
from config import PREDICTION_THRESHOLD

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

st.set_page_config(
    layout="wide",
    page_title="Anti Food Waste - Aalborg",
    page_icon="🛒",
)

# ── Cached data loading ────────────────────────────────────────────────────────

@st.cache_data(ttl=900)
def load_predictions() -> tuple[pd.DataFrame, str, str, str]:
    """Load pre-scored predictions from the MySQL app table. Cached for 15 minutes."""
    log.info("Cache miss — loading from MySQL app table")
    conn = get_connection()
    df   = pd.read_sql("SELECT * FROM app", conn)
    conn.close()
    log.info(f"Loaded {len(df)} scored offers from app table")

    if df.empty:
        return pd.DataFrame(), "", "", datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model_type = df["champion_model"].iloc[0]    if "champion_model"    in df.columns else "unknown"
    trained_on = df["champion_trained_on"].iloc[0] if "champion_trained_on" in df.columns else "unknown"
    fetched_at = df["predicted_at"].iloc[0]      if "predicted_at"      in df.columns else "unknown"

    df["store_lat"] = pd.to_numeric(df["store_lat"], errors="coerce")
    df["store_lng"] = pd.to_numeric(df["store_lng"], errors="coerce")
    df = df.sort_values("sell_probability", ascending=False).reset_index(drop=True)

    return df, model_type, trained_on, fetched_at


# ── UI helpers ─────────────────────────────────────────────────────────────────

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


def _render_card(row: pd.Series) -> None:
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
        st.markdown(f"🏪 {brand} · {store}")

        # Prices
        new_p  = row.get("offer_new_price", "")
        orig_p = row.get("offer_original_price", "")
        disc   = row.get("offer_percent_discount", "")
        unit   = row.get("offer_stock_unit", "")
        st.markdown(
            f"<span style='font-size:20px;font-weight:700'>{new_p} kr</span> "
            f"<span style='text-decoration:line-through;color:#888'>{orig_p} kr</span> "
            f"<span style='color:#e67e22'>-{disc}%</span>  "
            f"<span style='color:#999;font-size:12px'>/{unit}</span>",
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
    st.title("Anti Food Waste — Aalborg")
    st.markdown("*Live clearance offers ranked by sell-through probability*")

    try:
        results, model_type, trained_on, fetched_at = load_predictions()
    except Exception as exc:
        if "sql-net" in str(exc) or "Can't connect" in str(exc) or "Connection" in str(exc):
            st.error("Cannot connect to database — check UCloud sql-net connection")
        elif "model.joblib" in str(exc) or "scaler.joblib" in str(exc) or "label_encoders" in str(exc):
            st.warning("No trained model found — run the ML pipeline first")
        else:
            st.error(f"Error loading predictions: {exc}")
        st.stop()

    # Header row: last updated + refresh
    col_ts, col_btn = st.columns([4, 1])
    with col_ts:
        st.caption(f"Last updated: {fetched_at}  ·  Cache refreshes every 15 min")
    with col_btn:
        if st.button("Refresh now"):
            st.cache_data.clear()
            st.rerun()

    if results.empty:
        st.info("No clearance offers available right now. Check back later.")
        st.stop()

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
        sel_stores = st.multiselect(
            "Store name",
            options=all_stores,
            default=all_stores,
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

    # ── Summary metrics ────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total offers", len(df))
    m2.metric("Will sell", int(df["will_sell"].sum()))
    m3.metric("Won't sell", int((df["will_sell"] == 0).sum()))
    m4.metric("Avg probability", f"{df['sell_probability'].mean():.0%}" if len(df) else "—")

    st.divider()

    # ── Store map ──────────────────────────────────────────────────────────────
    with st.expander("Store map — hover a pin to see offers", expanded=True):
        geo = df.dropna(subset=["store_lat", "store_lng"])
        if geo.empty:
            st.caption("No location data available.")
        else:
            store_rows = []
            for (lat, lng, name, brand), grp in geo.groupby(
                ["store_lat", "store_lng", "store_name", "store_brand"], sort=False
            ):
                n_total = len(grp)
                n_sell  = int(grp["will_sell"].sum())
                sell_pct = n_sell / n_total if n_total else 0

                # Pin colour: green ≥50 % sell, orange 20-50 %, red <20 %
                if sell_pct >= 0.5:
                    color = [46, 204, 113, 230]
                elif sell_pct >= 0.2:
                    color = [230, 126, 34, 230]
                else:
                    color = [231, 76, 60, 230]

                # Build offer list for tooltip (top 6 by probability)
                lines = []
                for _, r in grp.sort_values("sell_probability", ascending=False).head(6).iterrows():
                    verdict = "✓" if r["will_sell"] == 1 else "✗"
                    lines.append(
                        f"{verdict} {r['product_description']} — "
                        f"{r['offer_new_price']} kr "
                        f"({int(r['sell_probability'] * 100)}%)"
                    )
                offers_html = "<br/>".join(lines)
                if n_total > 6:
                    offers_html += f"<br/><i>… and {n_total - 6} more</i>"

                store_rows.append({
                    "lat":        lat,
                    "lon":        lng,
                    "store_name": name,
                    "store_brand": _brand_label(brand),
                    "n_total":    n_total,
                    "n_sell":     n_sell,
                    "color":      color,
                    "offers_html": offers_html,
                })

            store_map_df = pd.DataFrame(store_rows)

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=store_map_df,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=80,
                radius_min_pixels=8,
                radius_max_pixels=24,
                pickable=True,
            )

            view = pdk.ViewState(
                latitude=store_map_df["lat"].mean(),
                longitude=store_map_df["lon"].mean(),
                zoom=12,
                pitch=0,
            )

            tooltip = {
                "html": (
                    "<b style='font-size:14px'>{store_name}</b> "
                    "<span style='color:#888'>({store_brand})</span><br/>"
                    "<b>{n_total}</b> offers &nbsp;·&nbsp; "
                    "<b style='color:#27ae60'>{n_sell} will sell</b>"
                    "<hr style='margin:6px 0;border-color:#ddd'/>"
                    "{offers_html}"
                ),
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
                    layers=[layer],
                    initial_view_state=view,
                    tooltip=tooltip,
                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                ),
                use_container_width=True,
            )
            st.caption("🟢 ≥50% sell  🟠 20–50%  🔴 <20%")

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
                _render_card(row)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.caption(f"Model: {model_type}  ·  Trained: {trained_on}  ·  Threshold: {PREDICTION_THRESHOLD}")


if __name__ == "__main__":
    main()
