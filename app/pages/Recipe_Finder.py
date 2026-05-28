import re
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# app/pages/Recipe_Finder.py -> parent.parent.parent = project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "llm_dev"))

import json

import streamlit as st
from fpdf import FPDF

# ── Pipeline imports — all logic lives in llm_dev/query.py ───────────────────
from query import (
    _chroma_client,
    retrieve_recipes,
    fetch_current_products,
    get_top_products_for_recipe,
    format_recipes_for_llm,
    format_products_per_recipe,
    load_prompt,
    call_llm,
    TOP_K_RECIPES as TOP_K,
)

# ── Constants ─────────────────────────────────────────────────────────────────
CHROMA_PATH = str(PROJECT_ROOT / "data/chroma_db")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Recipe Finder — Anti Food Waste",
    page_icon="🍽️",
)

# ── Cached pipeline wrappers ──────────────────────────────────────────────────

@st.cache_resource
def get_chroma():
    return _chroma_client()


@st.cache_data(ttl=900)
def cached_fetch_current_products() -> dict:
    return fetch_current_products()


# ── Time helpers ──────────────────────────────────────────────────────────────

def _parse_time_to_minutes(time_str: str) -> int | None:
    """Parse Danish duration strings like '45 min', '1 time', '1 time 30 min'."""
    if not time_str:
        return None
    total = 0
    h = re.search(r"(\d+)\s*time", time_str)
    m = re.search(r"(\d+)\s*min", time_str)
    if h:
        total += int(h.group(1)) * 60
    if m:
        total += int(m.group(1))
    return total if total > 0 else None


def _base_servings(servings_str: str) -> int:
    """Extract the first number from a servings string, defaulting to 4."""
    match = re.search(r"\d+", servings_str or "")
    return int(match.group()) if match else 4


# ── Pipeline ──────────────────────────────────────────────────────────────────

def filter_by_time(recipes: list[dict], max_minutes: int | None) -> list[dict]:
    """Keep only recipes within the time budget (None = no limit), ranked by relevance."""
    if max_minutes is None:
        return recipes[:TOP_K]
    filtered = []
    for r in recipes:
        t = _parse_time_to_minutes(r.get("total_time", ""))
        if t is None or t <= max_minutes:
            filtered.append(r)
    return filtered[:TOP_K]




def parse_llm_sections(text: str) -> list[str]:
    """Parse JSON response into 3 ingredient strings.

    Expected format: {"opskrift_1": "...", "opskrift_2": "...", "opskrift_3": "..."}
    Falls back to regex splitting if JSON parsing fails.
    """
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            sections = [
                str(data.get("opskrift_1", "")).strip(),
                str(data.get("opskrift_2", "")).strip(),
                str(data.get("opskrift_3", "")).strip(),
            ]
        elif isinstance(data, list):
            sections = [str(s).strip() for s in data[:3]]
        else:
            sections = []
    except (json.JSONDecodeError, TypeError):
        # Regex fallback in case the model ignores JSON mode
        parts    = re.split(r"={2,}\s*OPSKRIFT[_\s]?\d+\s*={2,}", text, flags=re.IGNORECASE)
        sections = [p.strip() for p in parts[1:4]]
    while len(sections) < 3:
        sections.append("")
    return sections


# ── PDF helpers ───────────────────────────────────────────────────────────────

def _pdf_safe(text: str) -> str:
    """Replace characters outside latin-1 with safe ASCII equivalents."""
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
    servings_label: str,
    ingredients_md: str,
    instructions: list[dict],
    recipe_url: str = "",
) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(20, 20, 20)
    w = pdf.epw

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(w, 10, _pdf_safe(title))
    pdf.ln(2)

    # Source URL
    if recipe_url:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 100, 100)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(w, 5, recipe_url)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    # Description
    if description:
        pdf.set_font("Helvetica", "", 10)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(w, 6, _pdf_safe(description))
        pdf.ln(2)

    # Meta line: time + servings
    meta_parts = []
    if total_time:
        meta_parts.append(f"Tilberedningstid: {total_time}")
    if servings_label:
        meta_parts.append(servings_label)
    if meta_parts:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(100, 100, 100)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(w, 6, _pdf_safe("  |  ".join(meta_parts)))
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)

    # Ingredients
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

    # Instructions
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
    """Colour-code substitution lines.

    LLM format: - 200 g smør -> [TILBUD] LURPAK SMØR, Netto Aalborg, 22,00 kr (30% rabat)
    Original ingredient stays normal; -> [TILBUD] part is green.
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

    # ── Sidebar filters ────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Filters")

        desired_servings = st.slider(
            "Antal portioner",
            min_value=1,
            max_value=8,
            value=4,
            step=1,
        )

        st.divider()

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
                # Over-fetch candidates when a time filter is active so we have
                # enough to pick 3 after filtering.
                n_candidates     = TOP_K * 4 if max_minutes else TOP_K
                candidates       = retrieve_recipes(user_query, n_candidates)
                recipes          = filter_by_time(candidates, max_minutes)
                current_products = cached_fetch_current_products()
            except Exception as exc:
                st.error(f"Could not load data: {exc}")
                return

            if not recipes:
                st.warning(
                    "No recipes found within the time limit — try relaxing the filter."
                    if max_minutes else
                    "No matching recipes found — try a different description."
                )
                return

            try:
                chroma = get_chroma()
                products_per_recipe = [
                    get_top_products_for_recipe(r["_id"], current_products, chroma)
                    for r in recipes
                ]
                system, user_template = load_prompt()
                user_message = user_template.format(
                    query=user_query,
                    recipes=format_recipes_for_llm(recipes, desired_servings),
                    products=format_products_per_recipe(recipes, products_per_recipe),
                )
                answer   = call_llm(system, user_message)
                sections = parse_llm_sections(answer)
            except Exception as exc:
                st.error(f"LLM call failed: {exc}")
                return

        st.session_state["recipe_results"] = {
            "recipes":          recipes,
            "sections":         sections,
            "desired_servings": desired_servings,
        }

    # ── Display results ────────────────────────────────────────────────────────
    # Persists across reruns caused by download buttons.
    data = st.session_state.get("recipe_results")
    if not data:
        return

    recipes          = data["recipes"]
    sections         = data["sections"]
    desired_servings = data["desired_servings"]

    st.divider()
    st.subheader("Your recipes")

    cols = st.columns(len(recipes))
    for i, (col, recipe, ingredients) in enumerate(zip(cols, recipes, sections)):
        with col:
            # ── Image ──────────────────────────────────────────────────────────
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

            # ── Title ──────────────────────────────────────────────────────────
            recipe_url = recipe.get("url", "")
            title      = recipe.get("title", f"Opskrift {i + 1}")
            if recipe_url:
                st.markdown(f"### [{title}]({recipe_url})")
            else:
                st.markdown(f"### {title}")

            # ── Description ────────────────────────────────────────────────────
            description = recipe.get("description", "")
            if description:
                st.markdown(description)

            # ── Meta: time + servings ──────────────────────────────────────────
            total_time = recipe.get("total_time", "")
            original_s = recipe.get("servings", "")
            base       = _base_servings(original_s)

            meta_cols = st.columns(2)
            with meta_cols[0]:
                if total_time:
                    st.caption(f"⏱️ {total_time}")
            with meta_cols[1]:
                if original_s:
                    if desired_servings != base:
                        st.caption(f"👥 {original_s} → **{desired_servings}**")
                    else:
                        st.caption(f"👥 {original_s}")

            st.divider()

            # ── Ingredients ────────────────────────────────────────────────────
            st.markdown("**Ingredienser**")
            _render_ingredients(ingredients)

            st.divider()

            # ── Walkthrough ────────────────────────────────────────────────────
            instructions = recipe.get("instructions", [])
            if instructions:
                with st.expander("Fremgangsmåde", expanded=False):
                    _render_instructions(instructions)

            # ── PDF download ───────────────────────────────────────────────────
            servings_label = ""
            if original_s:
                servings_label = (
                    f"Portioner: {desired_servings} (originalt: {original_s})"
                    if desired_servings != base else
                    f"Portioner: {original_s}"
                )

            st.download_button(
                label="Download som PDF",
                data=build_recipe_pdf(
                    title=title,
                    description=description,
                    total_time=total_time,
                    servings_label=servings_label,
                    ingredients_md=ingredients,
                    instructions=instructions,
                    recipe_url=recipe_url,
                ),
                file_name=f"{title[:40].replace(' ', '_')}.pdf",
                mime="application/pdf",
                key=f"pdf_{i}",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
