"""
Arla recipe scraper.

Fetches the recipe sitemap from arla.dk, then scrapes each recipe page
and saves the result as a JSON file (one file per recipe).

Usage:
    python recipe_scraper/scrape_recipes.py --limit 10
    python recipe_scraper/scrape_recipes.py --limit 0   # all recipes
"""

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

SITEMAP_URL = (
    "https://www.arla.dk/sitemap.xml"
    "?type=Modules.Recipes.Business.SitemapUrlWriter.RecipeSitemapUrlWriter"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; RecipeScraper/1.0; "
        "+https://github.com/ConvoTechDK)"
    )
}

COPENHAGEN_TZ = ZoneInfo("Europe/Copenhagen")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_duration(iso: str) -> str:
    """Convert ISO 8601 duration to a human-readable Danish string.

    'PT1H'    -> '1 time'
    'PT30M'   -> '30 min'
    'PT1H30M' -> '1 time 30 min'
    'PT2H'    -> '2 timer'
    'PT00M'   -> ''
    """
    if not iso:
        return ""
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?", iso.upper())
    if not m:
        return iso
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    if hours == 0 and minutes == 0:
        return ""
    parts = []
    if hours == 1:
        parts.append("1 time")
    elif hours > 1:
        parts.append(f"{hours} timer")
    if minutes > 0:
        parts.append(f"{minutes} min")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Sitemap
# ---------------------------------------------------------------------------

def fetch_sitemap(limit: int) -> list[tuple[str, str]]:
    """Return list of (recipe_url, image_url) from the Arla sitemap."""
    resp = requests.get(SITEMAP_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    ns = {
        "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
        "image": "http://www.google.com/schemas/sitemap-image/1.1",
    }

    root = ET.fromstring(resp.content)
    entries = []
    for url_el in root.findall("sm:url", ns):
        loc = url_el.findtext("sm:loc", namespaces=ns) or ""
        image_loc = url_el.findtext("image:image/image:loc", namespaces=ns) or ""
        if loc:
            entries.append((loc.strip(), image_loc.strip()))

    if limit > 0:
        entries = entries[:limit]

    return entries


# ---------------------------------------------------------------------------
# Recipe page parsing
# ---------------------------------------------------------------------------

def _extract_json_ld(soup: BeautifulSoup) -> dict:
    """Try to find a Schema.org Recipe object in any JSON-LD block."""
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        nodes = data if isinstance(data, list) else [data]
        for node in nodes:
            if isinstance(node, dict):
                if node.get("@type") == "Recipe":
                    return node
                for item in node.get("@graph", []):
                    if isinstance(item, dict) and item.get("@type") == "Recipe":
                        return item
    return {}


def _extract_microdata(soup: BeautifulSoup) -> dict:
    """Fall back to itemprop microdata attributes."""

    def first_text(prop: str) -> str:
        el = soup.find(itemprop=prop)
        if not el:
            return ""
        return el.get("content") or el.get_text(strip=True)

    def all_text(prop: str) -> list[str]:
        return [
            (el.get("content") or el.get_text(strip=True))
            for el in soup.find_all(itemprop=prop)
        ]

    result = {
        "name": first_text("name"),
        "description": first_text("description"),
        "recipeYield": first_text("recipeYield"),
        "prepTime": first_text("prepTime"),
        "cookTime": first_text("cookTime"),
        "totalTime": first_text("totalTime"),
        "recipeIngredient": all_text("recipeIngredient"),
        "recipeCategory": all_text("recipeCategory"),
        "keywords": first_text("keywords"),
    }

    instructions = []
    for el in soup.find_all(itemprop="recipeInstructions"):
        steps = el.find_all(itemprop="text")
        if steps:
            instructions.extend(s.get_text(strip=True) for s in steps)
        else:
            text = el.get_text(strip=True)
            if text:
                instructions.append(text)
    result["recipeInstructions"] = instructions

    nutrition_el = soup.find(itemprop="nutrition")
    if nutrition_el:

        def _nutr(prop: str) -> str:
            el = nutrition_el.find(itemprop=prop)
            return el.get_text(strip=True) if el else ""

        result["nutrition"] = {
            "calories": _nutr("calories"),
            "protein": _nutr("proteinContent"),
            "fat": _nutr("fatContent"),
            "carbohydrates": _nutr("carbohydrateContent"),
            "fiber": _nutr("fiberContent"),
        }

    return result


def _extract_instructions_html(soup: BeautifulSoup) -> list[dict]:
    """Find the 'Sådan gør du' section and extract instructions with sections.

    Arla pages use h2 'Sådan gør du' followed by flat h3 sub-section headings
    each paired with a <ul>/<li> step list, all as direct siblings.

    Returns a list of {"section": str|None, "steps": [str]} dicts.
    """
    target = None
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5"]):
        if "sådan" in tag.get_text().lower():
            target = tag
            break

    if not target:
        return []

    heading_level = int(target.name[1])
    sections: list[dict] = []
    current: dict = {"section": None, "steps": []}

    for sibling in target.find_next_siblings():
        if not sibling.name:
            continue
        if sibling.name[0] == "h" and sibling.name[1:].isdigit():
            level = int(sibling.name[1:])
            if level <= heading_level:
                break
            # New sub-section heading
            if current["steps"] or current["section"] is not None:
                sections.append(current)
            current = {"section": sibling.get_text(strip=True), "steps": []}
        elif sibling.name in ("ul", "ol"):
            for li in sibling.find_all("li"):
                text = li.get_text(strip=True)
                if text:
                    current["steps"].append({"step": len(current["steps"]) + 1, "text": text})
        elif sibling.name == "li":
            text = sibling.get_text(strip=True)
            if text:
                current["steps"].append({"step": len(current["steps"]) + 1, "text": text})

    if current["steps"] or current["section"] is not None:
        sections.append(current)

    return sections


def _normalise(raw: dict, soup: BeautifulSoup) -> dict:
    """Map raw Schema.org field names to the output document shape."""

    def _instruction_sections(instructions) -> list[dict]:
        """Convert JSON-LD recipeInstructions to [{section, steps}] format.

        Handles plain strings, HowToStep dicts, and HowToSection dicts.
        """
        if not instructions:
            return []
        if isinstance(instructions, str):
            return [{"section": None, "steps": [{"step": 1, "text": instructions}]}] if instructions else []

        sections: list[dict] = []
        current: dict = {"section": None, "steps": []}

        def _numbered(texts: list[str]) -> list[dict]:
            return [{"step": i + 1, "text": t} for i, t in enumerate(texts)]

        for item in instructions:
            if isinstance(item, str):
                current["steps"].append({"step": len(current["steps"]) + 1, "text": item})
            elif isinstance(item, dict):
                item_type = item.get("@type", "")
                if item_type == "HowToSection":
                    if current["steps"] or current["section"] is not None:
                        sections.append(current)
                    raw_steps = []
                    for step in item.get("itemListElement", []):
                        text = step.get("text", "") if isinstance(step, dict) else str(step)
                        if text:
                            raw_steps.append(text)
                    sections.append({"section": item.get("name"), "steps": _numbered(raw_steps)})
                    current = {"section": None, "steps": []}
                else:
                    text = item.get("text", "")
                    if text:
                        current["steps"].append({"step": len(current["steps"]) + 1, "text": text})

        if current["steps"] or current["section"] is not None:
            sections.append(current)

        return [s for s in sections if s["steps"]]

    def _nutrition(raw_nutrition) -> dict:
        if not raw_nutrition or not isinstance(raw_nutrition, dict):
            return {}
        return {
            "per": "100g",
            "calories": raw_nutrition.get("calories", ""),
            "protein": raw_nutrition.get("proteinContent", raw_nutrition.get("protein", "")),
            "fat": raw_nutrition.get("fatContent", raw_nutrition.get("fat", "")),
            "carbohydrates": raw_nutrition.get("carbohydrateContent", raw_nutrition.get("carbohydrates", "")),
            "fiber": raw_nutrition.get("fiberContent", raw_nutrition.get("fiber", "")),
        }

    keywords = raw.get("keywords", "")
    if isinstance(keywords, list):
        keyword_list = keywords
    elif isinstance(keywords, str):
        keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]
    else:
        keyword_list = []

    categories_raw = raw.get("recipeCategory", [])
    if isinstance(categories_raw, str):
        categories = [c.strip() for c in categories_raw.split(",") if c.strip()]
    elif isinstance(categories_raw, list):
        categories = []
        for c in categories_raw:
            categories.extend(p.strip() for p in str(c).split(",") if p.strip())
    else:
        categories = []

    instructions = _instruction_sections(raw.get("recipeInstructions", []))
    if not instructions:
        instructions = _extract_instructions_html(soup)

    # A single section with no meaningful name (e.g. Arla CMS placeholder
    # "First instruction") should have section set to null.
    if len(instructions) == 1:
        instructions[0]["section"] = None

    return {
        "title": raw.get("name", ""),
        "description": raw.get("description", ""),
        "servings": str(raw.get("recipeYield", "")),
        "prep_time": _parse_duration(raw.get("prepTime", "")),
        "cook_time": _parse_duration(raw.get("cookTime", "")),
        "total_time": _parse_duration(raw.get("totalTime", "")),
        "ingredients": raw.get("recipeIngredient", []),
        "instructions": instructions,
        "nutrition": _nutrition(raw.get("nutrition", {})),
        "categories": categories,
        "keywords": keyword_list,
    }


def scrape_recipe(url: str, image_url: str) -> dict:
    """Fetch and parse a single recipe page. Returns a complete document dict."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    raw = _extract_json_ld(soup)
    if not raw:
        raw = _extract_microdata(soup)
    if not raw.get("name"):
        h1 = soup.find("h1")
        if h1:
            raw["name"] = h1.get_text(strip=True)

    doc = _normalise(raw, soup)
    doc.update({
        "url": url,
        "image_url": image_url,
        "scraped_at": datetime.now(COPENHAGEN_TZ).isoformat(),
    })
    return doc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Arla recipes to JSON files.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of recipes to scrape (0 = all)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/recipes",
        help="Directory to write JSON files into (default: data/recipes)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between requests (default: 1.0)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching sitemap … (limit={args.limit or 'all'})")
    entries = fetch_sitemap(args.limit)
    total = len(entries)
    print(f"Found {total} recipe(s) to scrape.\n")

    ok = 0
    failed = 0

    for i, (url, image_url) in enumerate(entries, start=1):
        slug = url.rstrip("/").rsplit("/", 1)[-1] or f"recipe_{i}"
        out_path = output_dir / f"{slug}.json"

        try:
            doc = scrape_recipe(url, image_url)
            out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{i}/{total}] OK  {doc['title'] or slug}")
            ok += 1
        except Exception as exc:
            err_doc = {
                "url": url,
                "error": str(exc),
                "scraped_at": datetime.now(COPENHAGEN_TZ).isoformat(),
            }
            out_path.write_text(json.dumps(err_doc, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{i}/{total}] FAIL {slug} — {exc}", file=sys.stderr)
            failed += 1

        if i < total:
            time.sleep(args.delay)

    print(f"\nDone. {ok} scraped, {failed} failed. Files in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
