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


def parse_single_hours(entry: dict) -> str | None:
    """
    Format a single store hours entry into a readable string.
    Example output: "07:00-22:00" or "closed"
    """
    if not entry:
        return None
    if entry.get("closed", False):
        return "closed"
    # Slice just the HH:MM portion from the full ISO timestamp e.g. "2024-01-01T07:00:00"
    open_time  = entry.get("open",  "")[-8:-3]
    close_time = entry.get("close", "")[-8:-3]
    return f"{open_time}-{close_time}"


def parse_hours_today_tomorrow(hours: list[dict]) -> tuple[str | None, str | None]:
    """
    Split the store hours array into today and tomorrow.
    The API always returns exactly two entries: index 0 = today, index 1 = tomorrow.
    Returns a tuple of (today_hours, tomorrow_hours) as readable strings.
    """
    today    = parse_single_hours(hours[0] if len(hours) > 0 else {})
    tomorrow = parse_single_hours(hours[1] if len(hours) > 1 else {})
    return today, tomorrow


def parse_customer_flow(hours: list[dict]) -> tuple[str | None, str | None]:
    """
    Extract customerFlow arrays separately for today and tomorrow.
    customerFlow is a list of numbers representing store busyness per hour.
    The API returns index 0 = today, index 1 = tomorrow.
    Each is stored as a comma-separated string e.g. "0,5,12,8,3".
    Returns a tuple of (today_flow, tomorrow_flow).
    """
    def extract_flow(entry: dict) -> str | None:
        flow = entry.get("customerFlow", [])
        return ",".join(str(v) for v in flow) if flow else None

    today    = extract_flow(hours[0] if len(hours) > 0 else {})
    tomorrow = extract_flow(hours[1] if len(hours) > 1 else {})
    return today, tomorrow


def format_timestamp(raw: str | None) -> str | None:
    """
    Convert an API ISO 8601 timestamp to a Darts-compatible datetime string.

    API format:   "2019-11-15T22:23:23.000Z"  (UTC, with T, milliseconds, and Z)
    Darts format: "2019-11-15 22:23:23"        (timezone-naive, space separator)

    Darts requires timezone-naive timestamps so it can build a clean DatetimeIndex.
    We drop the timezone (Z = UTC) here — just be consistent and always fetch
    at the same timezone when building your time series later.
    """
    if not raw or raw == "N/A":
        return None
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

        # Split customerFlow into today and tomorrow separately
        store_customer_flow_today, store_customer_flow_tomorrow = parse_customer_flow(hours)

        # --- Loop over each clearance offer in this store ---
        for item in store.get("clearances", []):
            offer   = item.get("offer",   {})
            product = item.get("product", {})

            # Categories is a nested dict with "da" and "en" keys
            categories = product.get("categories", {})

            rows.append({
                # Product
                "product_ean":          product.get("ean",         None),
                "product_description":  product.get("description", None),
                "product_image":        product.get("image",       None),
                "product_category_da":  categories.get("da",       None),
                "product_category_en":  categories.get("en",       None),

                # Offer — timestamps formatted for Darts (timezone-naive ISO 8601)
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

                # Store
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


# --- Quick sanity check when running the file directly ---
if __name__ == "__main__":
    rows = fetch_food_waste()
    print(f"Fetched {len(rows)} clearance offer(s)\n")
    if rows:
        print("First row:")
        for key, value in rows[0].items():
            print(f"  {key}: {value}")