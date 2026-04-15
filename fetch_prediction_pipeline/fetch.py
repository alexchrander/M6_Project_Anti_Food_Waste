from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path so config.py is found
sys.path.append(str(Path(__file__).parent.parent))

import os
from typing import Any

import requests
from dotenv import load_dotenv

from config import ZIP_CODES, BASE_URL

load_dotenv()


def build_urls() -> list[str]:
    """Build one Salling Food Waste API URL per zip code."""
    return [f"{BASE_URL}?zip={zip_code}" for zip_code in ZIP_CODES]


def parse_single_hours(entry: dict) -> str | None:
    """
    Format a single store hours entry into a readable string.
    Example output: "07:00-22:00" or "closed"
    """
    if not entry:
        return None
    if entry.get("closed", False):
        return "closed"
    open_time  = entry.get("open",  "")[-8:-3]
    close_time = entry.get("close", "")[-8:-3]
    return f"{open_time}-{close_time}"


def parse_hours_today_tomorrow(hours: list[dict]) -> tuple[str | None, str | None]:
    """
    Split the store hours array into today and tomorrow.
    The API always returns exactly two entries: index 0 = today, index 1 = tomorrow.
    """
    today    = parse_single_hours(hours[0] if len(hours) > 0 else {})
    tomorrow = parse_single_hours(hours[1] if len(hours) > 1 else {})
    return today, tomorrow


def parse_customer_flow(hours: list[dict]) -> tuple[str | None, str | None]:
    """
    Extract customerFlow arrays separately for today and tomorrow.
    Each is stored as a comma-separated string e.g. "0,5,12,8,3".
    """
    def extract_flow(entry: dict) -> str | None:
        flow = entry.get("customerFlow", [])
        return ",".join(str(v) for v in flow) if flow else None

    today    = extract_flow(hours[0] if len(hours) > 0 else {})
    tomorrow = extract_flow(hours[1] if len(hours) > 1 else {})
    return today, tomorrow


def format_timestamp(raw: str | None) -> str | None:
    """
    Convert an API ISO 8601 timestamp to a timezone-naive datetime string.
    "2019-11-15T22:23:23.000Z" -> "2019-11-15 22:23:23"
    """
    if not raw or raw == "N/A":
        return None
    return raw.replace("T", " ")[:19]


def fetch_food_waste() -> list[dict[str, Any]]:
    """
    Fetch discounted near-expiry food items from the Salling API.
    Loops over all zip codes in ZIP_CODES and combines results.
    Returns a flat list of rows — one row per clearance offer.
    """
    api_key = os.getenv("ANTI_FOOD_WASTE_API")
    if not api_key:
        raise RuntimeError(
            "Missing ANTI_FOOD_WASTE_API. Add it to your .env file locally "
            "or as a GitHub Secret in CI."
        )

    headers = {"Authorization": f"Bearer {api_key}"}
    rows: list[dict[str, Any]] = []

    for url in build_urls():
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        for store in data:
            store_info  = store.get("store", {})
            address     = store_info.get("address", {})
            coordinates = store_info.get("coordinates", [None, None])
            hours       = store_info.get("hours", [])

            store_lng = coordinates[0] if len(coordinates) > 0 else None
            store_lat = coordinates[1] if len(coordinates) > 1 else None

            store_hours_today, store_hours_tomorrow             = parse_hours_today_tomorrow(hours)
            store_customer_flow_today, store_customer_flow_tomorrow = parse_customer_flow(hours)

            for item in store.get("clearances", []):
                offer      = item.get("offer",   {})
                product    = item.get("product", {})
                categories = product.get("categories", {})

                rows.append({
                    "product_ean":          product.get("ean",         None),
                    "product_description":  product.get("description", None),
                    "product_image":        product.get("image",       None),
                    "product_category_da":  categories.get("da",       None),
                    "product_category_en":  categories.get("en",       None),
                    "offer_ean":              offer.get("ean",             None),
                    "offer_currency":         offer.get("currency",        None),
                    "offer_original_price":   offer.get("originalPrice",   None),
                    "offer_new_price":        offer.get("newPrice",        None),
                    "offer_discount":         offer.get("discount",        None),
                    "offer_percent_discount": offer.get("percentDiscount", None),
                    "offer_stock":            offer.get("stock",           None),
                    "offer_stock_unit":       offer.get("stockUnit",       None),
                    "offer_start_time":       format_timestamp(offer.get("startTime",  None)),
                    "offer_end_time":         format_timestamp(offer.get("endTime",    None)),
                    "offer_last_update":      format_timestamp(offer.get("lastUpdate", None)),
                    "store_id":                      store_info.get("id",    None),
                    "store_name":                    store_info.get("name",  None),
                    "store_brand":                   store_info.get("brand", None),
                    "store_lat":                     store_lat,
                    "store_lng":                     store_lng,
                    "store_street":                  address.get("street",  None),
                    "store_city":                    address.get("city",    None),
                    "store_zip":                     address.get("zip",     None),
                    "store_country":                 address.get("country", None),
                    "store_hours_today":             store_hours_today,
                    "store_hours_tomorrow":          store_hours_tomorrow,
                    "store_customer_flow_today":     store_customer_flow_today,
                    "store_customer_flow_tomorrow":  store_customer_flow_tomorrow,
                })

    return rows


if __name__ == "__main__":
    rows = fetch_food_waste()
    print(f"Fetched {len(rows)} clearance offer(s)\n")
    if rows:
        print("First row:")
        for key, value in rows[0].items():
            print(f"  {key}: {value}")