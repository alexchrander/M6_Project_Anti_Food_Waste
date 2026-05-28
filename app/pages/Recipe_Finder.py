import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "llm_pipeline"))

import streamlit as st
from fpdf import FPDF

from query import (
    _chroma_client,
    fetch_active_products,
    run_recipe_pipeline,
)
from maps_utils import geocode, haversine_km

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
    ingredients_md: str,
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

    for line in ingredients_md.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("- ") or line.startswith("* "):
            line = line[2:]
        is_sub = "[TILBUD]" in line.upper()
        if is_sub:
            pdf.set_text_color(39, 174, 96)
            pdf.set_font("Helvetica", "B", 10)
        else:
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 10)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(w, 6, _pdf_safe(line))

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

def _render_ingredients(ingredients: str) -> None:
    """Colour-code substitution lines green.

    LLM format: - 200 g smør -> [TILBUD] LURPAK SMØR, Netto Aalborg, 22,00 kr
    """
    if not ingredients:
        st.caption("Ingen ingredienser fundet.")
        return

    html_lines = []
    for line in ingredients.splitlines():
        stripped = line.strip().lstrip("- *")
        if not stripped:
            continue
        if "[TILBUD]" in stripped.upper() and "->" in stripped:
            orig, sub = stripped.split("->", 1)
            html_lines.append(
                f"{orig.strip()} "
                f"<span style='color:#27ae60;font-weight:600'>&#8594; {sub.strip()}</span>"
            )
        elif "[TILBUD]" in stripped.upper():
            html_lines.append(
                f"<span style='color:#27ae60;font-weight:600'>{stripped}</span>"
            )
        else:
            html_lines.append(stripped)

    st.markdown("<br>".join(html_lines), unsafe_allow_html=True)


def _render_instructions(instructions: list[dict]) -> None:
    for section in instructions:
        if section.get("section"):
            st.markdown(f"**{section['section']}**")
        for step in section.get("steps", []):
            st.markdown(f"{step['step']}. {step['text']}")


# ── Main UI ───────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🍽️ Recipe Finder")
    st.markdown(
        "Tell us what you feel like eating and we'll suggest **3 matching recipes** "
        "— with ingredients swapped for today's discounted offers where possible."
    )

    # ── Shared location input ──────────────────────────────────────────────────
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
        with st.spinner("Finding recipes and checking today's offers…"):
            try:
                active_products = cached_fetch_active_products()

                # Filter to nearby stores if location + distance are set
                user_location = st.session_state.get("user_location")
                dist_unit     = st.session_state.get("dist_unit", "km")
                max_dist      = st.session_state.get("max_dist_km" if dist_unit == "km" else "max_dist_min", None)
                if user_location and max_dist is not None:
                    speed_kmh = {"walking": 5, "bicycling": 15, "driving": 40, "transit": 25}.get(
                        st.session_state.get("travel_mode", "walking"), 5
                    )
                    max_km = max_dist if dist_unit == "km" else (max_dist / 60) * speed_kmh
                    active_products = {
                        ean: p for ean, p in active_products.items()
                        if p.get("store_lat") and p.get("store_lng")
                        and haversine_km(
                            user_location["lat"], user_location["lng"],
                            float(p["store_lat"]), float(p["store_lng"]),
                        ) <= max_km
                    }

                recipes, sections = run_recipe_pipeline(
                    query           = user_query,
                    chroma          = get_chroma(),
                    max_minutes     = max_minutes,
                    active_products = active_products,
                )
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
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
    for i, (col, recipe, ingredients) in enumerate(zip(cols, recipes, sections)):
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
            _render_ingredients(ingredients)

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
                    ingredients_md = ingredients,
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
