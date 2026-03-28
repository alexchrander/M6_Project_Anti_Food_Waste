from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

# Import our constants from config
from config import ZIP_CODE, BASE_URL

# Load environment variables from the .env file
load_dotenv()


def build_url() -> str:
    """
    Build the Salling Food Waste API URL using zip code from config.
    """
    return f"{BASE_URL}?zip={ZIP_CODE}"


def parse_single_hours(entry: dict) -> str:
    """
    Format a single store hours entry into a readable string.
    Example output: "07:00-22:00" or "closed"
    """
    if not entry:
        return "N/A"
    if entry.get("closed", False):
        return "closed"
    # Slice just the HH:MM portion from the full ISO timestamp e.g. "2024-01-01T07:00:00"
    open_time  = entry.get("open",  "")[-8:-3]
    close_time = entry.get("close", "")[-8:-3]
    return f"{open_time}-{close_time}"


def parse_hours_today_tomorrow(hours: list[dict]) -> tuple[str, str]:
    """
    Split the store hours array into today and tomorrow.
    The API always returns exactly two entries: index 0 = today, index 1 = tomorrow.
    Returns a tuple of (today_hours, tomorrow_hours) as readable strings.
    """
    today    = parse_single_hours(hours[0] if len(hours) > 0 else {})
    tomorrow = parse_single_hours(hours[1] if len(hours) > 1 else {})
    return today, tomorrow


def format_timestamp(raw: str) -> str:
    """
    Convert an API ISO 8601 timestamp to a Darts-compatible datetime string.

    API format:   "2019-11-15T22:23:23.000Z"  (UTC, with T, milliseconds, and Z)
    Darts format: "2019-11-15 22:23:23"        (timezone-naive, space separator)

    Darts requires timezone-naive timestamps so it can build a clean DatetimeIndex.
    We drop the timezone (Z = UTC) here — just be consistent and always fetch
    at the same timezone when building your time series later.
    """
    if not raw or raw == "N/A":
        return "N/A"
    # Replace the "T" separator with a space, then cut off milliseconds and "Z"
    # "2019-11-15T22:23:23.000Z" -> "2019-11-15 22:23:23"
    return raw.replace("T", " ")[:19]


def fetch_food_waste() -> list[dict[str, Any]]:
    """
    Fetch discounted near-expiry food items from the Salling API.
    Returns a flat list of rows — one row per clearance offer —
    with all store, offer, and product fields included.
    """
    # Read the API token from environment — never hardcode secrets!
    api_key = os.getenv("ANTI_FOOD_WASTE_API")
    if not api_key:
        raise RuntimeError(
            "Missing ANTI_FOOD_WASTE_API. Add it to your .env file locally "
            "or as a GitHub Secret in CI."
        )

    url = build_url()

    # Bearer token authentication in the request header
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    # Make the GET request with a timeout to avoid hanging forever
    response = requests.get(url, headers=headers, timeout=30)

    # Raise an exception automatically on 4xx/5xx status codes
    response.raise_for_status()

    # Parse the JSON response body
    data = response.json()

    # Build a flat list — one entry per clearance offer across all stores
    rows: list[dict[str, Any]] = []

    for store in data:
        # --- Store fields ---
        store_info  = store.get("store", {})
        address     = store_info.get("address", {})
        coordinates = store_info.get("coordinates", [None, None])
        hours       = store_info.get("hours", [])

        # Coordinates come as [longitude, latitude] from the API
        store_lng = coordinates[0] if len(coordinates) > 0 else None
        store_lat = coordinates[1] if len(coordinates) > 1 else None

        # Split opening hours into two separate columns: today and tomorrow
        store_hours_today, store_hours_tomorrow = parse_hours_today_tomorrow(hours)

        # --- Loop over each clearance offer in this store ---
        for item in store.get("clearances", []):
            offer   = item.get("offer",   {})
            product = item.get("product", {})

            rows.append({
                # Product
                "product_ean":         product.get("ean",         "N/A"),
                "product_description": product.get("description", "N/A"),
                "product_image":       product.get("image",       "N/A"),

                # Offer — timestamps formatted for Darts (timezone-naive ISO 8601)
                "offer_ean":              offer.get("ean",             "N/A"),
                "offer_currency":         offer.get("currency",        "N/A"),
                "offer_original_price":   offer.get("originalPrice",   "N/A"),
                "offer_new_price":        offer.get("newPrice",        "N/A"),
                "offer_discount":         offer.get("discount",        "N/A"),
                "offer_percent_discount": offer.get("percentDiscount", "N/A"),
                "offer_stock":            offer.get("stock",           "N/A"),
                "offer_stock_unit":       offer.get("stockUnit",       "N/A"),
                "offer_start_time":       format_timestamp(offer.get("startTime",  "N/A")),
                "offer_end_time":         format_timestamp(offer.get("endTime",    "N/A")),
                "offer_last_update":      format_timestamp(offer.get("lastUpdate", "N/A")),

                # Store
                "store_id":             store_info.get("id",    "N/A"),
                "store_name":           store_info.get("name",  "N/A"),
                "store_brand":          store_info.get("brand", "N/A"),
                "store_lat":            store_lat,
                "store_lng":            store_lng,
                "store_street":         address.get("street",  "N/A"),
                "store_city":           address.get("city",    "N/A"),
                "store_zip":            address.get("zip",     "N/A"),
                "store_country":        address.get("country", "N/A"),
                "store_hours_today":    store_hours_today,
                "store_hours_tomorrow": store_hours_tomorrow,
            })

    return rows


# --- Quick sanity check when running the file directly ---
if __name__ == "__main__":
    rows = fetch_food_waste()
    print(f"Fetched {len(rows)} clearance offer(s)\n")
    if rows:
        print("First row:")
        for key, value in rows[0].items():
            print(f"  {key}: {value}")