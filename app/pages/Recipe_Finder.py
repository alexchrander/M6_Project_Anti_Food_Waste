import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "rag_pipeline"))

import streamlit as st
from fpdf import FPDF

from query import (
    _chroma_client,
    fetch_active_products,
    run_recipe_pipeline,
)
from maps_utils import geocode, haversine_km, routes_cached
from config import OUTPUTS_DIR

# ── Page config ───────────────────────────────────────────────────────────────

# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def geocode_cached(address: str) -> dict:
    return geocode(address)


@st.cache_resource
def get_chroma():
    return _chroma_client()


@st.cache_data(ttl=900)
def cached_fetch_active_products() -> dict:
    return fetch_active_products()


# ── PDF generation ────────────────────────────────────────────────────────────

def _pdf_safe(text: str) -> str:
    return (
        text
        .replace("½", "1/2")
        .replace("¼", "1/4")
        .replace("¾", "3/4")
        .encode("latin-1", errors="replace")
        .decode("latin-1")
    )


def build_recipe_pdf(
    title: str,
    description: str,
    total_time: str,
    servings: str,
    ingredients,
    instructions: list[dict],
    recipe_url: str = "",
) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(20, 20, 20)
    w = pdf.epw

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(w, 10, _pdf_safe(title))
    pdf.ln(2)

    if recipe_url:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 100, 100)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(w, 5, recipe_url)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    if description:
        pdf.set_font("Helvetica", "", 10)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(w, 6, _pdf_safe(description))
        pdf.ln(2)

    meta_parts = []
    if total_time:
        meta_parts.append(f"Tilberedningstid: {total_time}")
    if servings:
        meta_parts.append(f"Portioner: {servings}")
    if meta_parts:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(100, 100, 100)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(w, 6, _pdf_safe("  |  ".join(meta_parts)))
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_x(pdf.l_margin)
    pdf.cell(w, 8, "Ingredienser", ln=True)
    pdf.ln(1)

    for ing in ingredients:
        if ing.substitution:
            sub = ing.substitution
            price_str = f"{sub.price:.2f} kr".replace(".", ",")
            line = f"{ing.text} -> {sub.product_name}, {sub.store}, {price_str} ({sub.discount_percent:.0f}% rabat)"
            pdf.set_text_color(39, 174, 96)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(w, 6, _pdf_safe(line))
            if sub.note:
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(w, 5, _pdf_safe(f"  ({sub.note})"))
        else:
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(w, 6, _pdf_safe(ing.text))

    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    if instructions:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_x(pdf.l_margin)
        pdf.cell(w, 8, "Fremgangsmade", ln=True)
        pdf.ln(1)
        for section in instructions:
            if section.get("section"):
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(w, 6, _pdf_safe(section["section"]))
            pdf.set_font("Helvetica", "", 10)
            for step in section.get("steps", []):
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(w, 6, _pdf_safe(f"{step['step']}. {step['text']}"))
            pdf.ln(2)

    return bytes(pdf.output())


# ── UI helpers ────────────────────────────────────────────────────────────────

def _render_ingredients(ingredients) -> None:
    """Render a list of Ingredient objects, colouring substitutions green."""
    if not ingredients:
        st.caption("Ingen ingredienser fundet.")
        return

    html_lines = []
    for ing in ingredients:
        if ing.substitution:
            sub = ing.substitution
            price_str = f"{sub.price:.2f} kr".replace(".", ",")
            sub_text = f"{sub.product_name}, {sub.store}, {price_str} ({sub.discount_percent:.0f}% rabat)"
            html_lines.append(
                f"{ing.text} "
                f"<span style='color:#27ae60;font-weight:600'>&#8594; {sub_text}</span>"
            )
            if sub.note:
                html_lines.append(
                    f"<span style='color:#27ae60;font-style:italic;font-size:0.9em'>"
                    f"({sub.note})</span>"
                )
        else:
            html_lines.append(ing.text)

    st.markdown("<br>".join(html_lines), unsafe_allow_html=True)


def _render_instructions(instructions: list[dict]) -> None:
    for section in instructions:
        if section.get("section"):
            st.markdown(f"**{section['section']}**")
        for step in section.get("steps", []):
            st.markdown(f"{step['step']}. {step['text']}")


# ── Logging ───────────────────────────────────────────────────────────────────

def _active_prompt_id() -> str:
    try:
        with open(PROJECT_ROOT / "rag_pipeline" / "prompt.json", encoding="utf-8") as f:
            return json.load(f)["active"]
    except Exception:
        return ""


def log_run(summary: dict) -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    log_path = OUTPUTS_DIR / "recipe_log.csv"
    file_exists = log_path.is_file()
    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary)


# ── Main UI ───────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🍽️ Recipe Finder")
    st.markdown(
        "Tell us what you feel like eating and we'll suggest **3 matching recipes**"
        "— automatically substituting ingredients with **clearance offers from Salling Group "
        "stores in Aalborg (9000)** to help reduce food waste."
    )

    st.info(
        "**This app only covers Aalborg (postal code 9000).**  "
        "Offers are fetched exclusively from **Salling Group** stores "
        "(Netto, Bilka & Føtex) located in Aalborg.  \n"
        "Please enter a **9000 Aalborg address** below to find nearby stores.",
        icon="📍",
    )

    # ── Shared location input ──────────────────────────────────────────────────
    loc_col, btn_col = st.columns([5, 1])
    with loc_col:
        address_input = st.text_input(
            "📍 Your location (9000 Aalborg)",
            value=st.session_state.get("user_address", ""),
            placeholder="e.g. Boulevarden 13, 9000 Aalborg or Nørregade 10, 9000 Aalborg",
            label_visibility="collapsed",
        )
    with btn_col:
        locate_btn = st.button("Locate", use_container_width=True)

    if locate_btn and address_input:
        try:
            loc = geocode_cached(address_input)
            formatted = loc.get("formatted_address", "")
            if "9000" not in formatted.lower() and "aalborg" not in formatted.lower() \
               and "9000" not in address_input.lower() and "aalborg" not in address_input.lower():
                st.warning(
                    "That address doesn't appear to be in Aalborg (9000). "
                    "This app only covers stores in Aalborg - please enter a 9000 Aalborg address."
                )
                st.session_state.pop("user_location", None)
            else:
                st.session_state["user_address"]  = address_input
                st.session_state["user_location"] = loc
        except Exception as exc:
            st.warning(f"Could not find that address: {exc}")
            st.session_state.pop("user_location", None)
    elif not address_input:
        st.session_state.pop("user_location", None)

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

    if st.session_state.get("user_location"):
        st.caption(f"📍 {st.session_state['user_location']['formatted_address']}")

    # ── Sidebar filters ────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Filters")

        no_time_limit = st.checkbox("Ingen tidsbegrænsning", value=True)
        if no_time_limit:
            max_minutes = None
        else:
            max_minutes = st.slider(
                "Max tilberedningstid",
                min_value=5,
                max_value=120,
                value=60,
                step=5,
                format="%d min",
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

    st.divider()

    # ── Query input ────────────────────────────────────────────────────────────
    user_query = st.text_area(
        "What would you like to cook?",
        placeholder="e.g. noget italiensk med pasta, en hurtig kyllingeret, vegetarisk aftensmad …",
        height=80,
    ).strip()

    find_btn = st.button("Find recipes", type="primary", disabled=not user_query)

    # ── Run pipeline ───────────────────────────────────────────────────────────
    if find_btn:
        user_location = st.session_state.get("user_location")
        dist_unit     = st.session_state.get("dist_unit", "km")
        max_dist      = st.session_state.get("max_dist_km" if dist_unit == "km" else "max_dist_min", None)
        location_filter_applied = bool(user_location and max_dist is not None)

        start_time = time.time()
        summary = {
            "timestamp":                  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "query":                      user_query,
            "duration_seconds":           None,
            "active_products_in_pool":    None,
            "nearby_stores":              "",
            "recipes_found":              None,
            "recipe_1_title":             "",
            "recipe_2_title":             "",
            "recipe_3_title":             "",
            "prompt_id":                             _active_prompt_id(),
            "recipe_1_substitutions_made":           None,
            "recipe_1_candidate_eans":               "",
            "recipe_1_candidate_descriptions":       "",
            "recipe_2_substitutions_made":           None,
            "recipe_2_candidate_eans":               "",
            "recipe_2_candidate_descriptions":       "",
            "recipe_3_substitutions_made":           None,
            "recipe_3_candidate_eans":               "",
            "recipe_3_candidate_descriptions":       "",
            "location_filter_applied":    location_filter_applied,
            "dist_unit":                  dist_unit if location_filter_applied else "",
            "max_dist":                   max_dist if location_filter_applied else "",
            "user_address":               address_input if location_filter_applied else "",
            "max_minutes":                max_minutes if max_minutes is not None else "",
            "pipeline_status":            "failed",
            "failed_step":                "",
        }

        with st.spinner("Finding recipes and checking today's offers…"):
            try:
                active_products = cached_fetch_active_products()

                # Filter to nearby stores if location + distance are set
                if location_filter_applied:
                    travel_mode = st.session_state.get("travel_mode", "walking")
                    speed_kmh = {"walking": 5, "bicycling": 15, "driving": 40, "transit": 25}.get(travel_mode, 5)
                    max_km = max_dist if dist_unit == "km" else (max_dist / 60) * speed_kmh

                    # Stage 1: haversine pre-filter with 1.5x buffer to limit route API calls
                    active_products = {
                        ean: p for ean, p in active_products.items()
                        if p.get("store_lat") and p.get("store_lng")
                        and haversine_km(
                            user_location["lat"], user_location["lng"],
                            float(p["store_lat"]), float(p["store_lng"]),
                        ) <= max_km * 1.5
                    }

                    # Stage 2: route API post-filter using actual travel distance/time
                    store_coords = {
                        p["store_name"]: (float(p["store_lat"]), float(p["store_lng"]))
                        for p in active_products.values()
                        if p.get("store_name") and p.get("store_lat") and p.get("store_lng")
                    }
                    if store_coords:
                        dest_key = tuple(
                            (name, lat, lng) for name, (lat, lng) in store_coords.items()
                        )
                        try:
                            routes = routes_cached(
                                user_location["formatted_address"], dest_key, travel_mode
                            )
                            if dist_unit == "km":
                                passing = {
                                    r["store_name"] for r in routes
                                    if r.get("distance_meters") and r["distance_meters"] / 1000 <= max_dist
                                }
                            else:
                                passing = {
                                    r["store_name"] for r in routes
                                    if r.get("duration_seconds") and r["duration_seconds"] / 60 <= max_dist
                                }
                            active_products = {
                                ean: p for ean, p in active_products.items()
                                if p.get("store_name") in passing
                            }
                        except Exception:
                            pass  # fall back to haversine pre-filtered set if routes API fails

                summary["active_products_in_pool"] = len(active_products)
                if location_filter_applied:
                    unique_stores = sorted(set(
                        p["store_name"]
                        for p in active_products.values()
                        if p.get("store_name")
                    ))
                    summary["nearby_stores"] = " | ".join(unique_stores)

                recipes, sections, products_per_recipe = run_recipe_pipeline(
                    query           = user_query,
                    chroma          = get_chroma(),
                    max_minutes     = max_minutes,
                    active_products = active_products,
                )

                summary["recipes_found"] = len(recipes)

                ean_by_name = {
                    p.get("product_description", ""): str(p["product_ean"])
                    for _, p in [pair for plist in products_per_recipe for pair in plist]
                }
                for i, (recipe, section, plist) in enumerate(zip(recipes, sections, products_per_recipe), start=1):
                    subs = [ing.substitution for ing in section.ingredients if ing.substitution is not None]
                    summary[f"recipe_{i}_title"]                    = recipe.get("title", "")
                    summary[f"recipe_{i}_substitutions_made"]       = len(subs)
                    summary[f"recipe_{i}_candidate_eans"]           = " | ".join(ean_by_name.get(s.product_name, "") for s in subs)
                    summary[f"recipe_{i}_candidate_descriptions"]   = " | ".join(s.product_name for s in subs)

                summary["pipeline_status"] = "success"

            except Exception as exc:
                summary["failed_step"] = "pipeline"
                st.error(f"Pipeline failed: {exc}")

            finally:
                summary["duration_seconds"] = round(time.time() - start_time, 2)
                log_run(summary)

        if summary["pipeline_status"] == "failed":
            return

        if not recipes:
            st.warning(
                "No recipes found within the time limit — try relaxing the filter."
                if max_minutes else
                "No matching recipes found — try a different description."
            )
            return

        st.session_state["recipe_results"] = {
            "recipes":  recipes,
            "sections": sections,
        }

    # ── Display results ────────────────────────────────────────────────────────
    data = st.session_state.get("recipe_results")
    if not data:
        return

    recipes  = data["recipes"]
    sections = data["sections"]

    st.divider()
    st.subheader("Your recipes")

    cols = st.columns(len(recipes))
    for i, (col, recipe, recipe_result) in enumerate(zip(cols, recipes, sections)):
        with col:
            img_url = recipe.get("image_url", "")
            if img_url:
                st.image(img_url, use_container_width=True)
            else:
                st.markdown(
                    "<div style='height:200px;background:#f0f0f0;border-radius:8px;"
                    "display:flex;align-items:center;justify-content:center;"
                    "color:#aaa;font-size:40px;'>🍽️</div>",
                    unsafe_allow_html=True,
                )

            recipe_url = recipe.get("url", "")
            title      = recipe.get("title", f"Opskrift {i + 1}")
            if recipe_url:
                st.markdown(f"### [{title}]({recipe_url})")
            else:
                st.markdown(f"### {title}")

            description = recipe.get("description", "")
            if description:
                st.markdown(description)

            total_time = recipe.get("total_time", "")
            servings   = recipe.get("servings", "")

            meta_cols = st.columns(2)
            with meta_cols[0]:
                if total_time:
                    st.caption(f"⏱️ {total_time}")
            with meta_cols[1]:
                if servings:
                    st.caption(f"👥 {servings}")

            st.divider()

            st.markdown("**Ingredienser**")
            _render_ingredients(recipe_result.ingredients)

            st.divider()

            instructions = recipe.get("instructions", [])
            if instructions:
                with st.expander("Fremgangsmåde", expanded=False):
                    _render_instructions(instructions)

            st.download_button(
                label="Download som PDF",
                data=build_recipe_pdf(
                    title        = title,
                    description  = description,
                    total_time   = total_time,
                    servings     = servings,
                    ingredients  = recipe_result.ingredients,
                    instructions = instructions,
                    recipe_url   = recipe_url,
                ),
                file_name=f"{title[:40].replace(' ', '_')}.pdf",
                mime="application/pdf",
                key=f"pdf_{i}",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
