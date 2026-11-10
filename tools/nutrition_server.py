"""nutrition-lookup: a small MCP server (stdio) exposing one tool, lookup_nutrition.

It searches USDA FoodData Central for generic foods (Foundation and SR Legacy data,
which report nutrients per 100 g) and scales calories/macros to the requested weight.
It returns the top few matches so the caller can judge which one fits.

Run standalone with: python tools/nutrition_server.py   (speaks MCP over stdio)
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import Field

SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
DATA_TYPES = ["Foundation", "SR Legacy"]
MAX_MATCHES = 5
TIMEOUT_SECONDS = 10
ATTEMPTS = 2
DNS_WARMUP_SECONDS = 120
RETRY_STATUSES = {502, 503, 504}
# USDA nutrient numbers. Foundation foods sometimes omit 208 and report Atwater
# energy (957/958) instead.
ENERGY_NUMBERS = ("208", "957", "958")
MACRO_NUMBERS = {"protein_g": "203", "fat_g": "204", "carbs_g": "205"}

# Logs go to stderr: stdout is the MCP transport.
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("nutrition-server")
logging.getLogger("httpx").setLevel(logging.WARNING)  # no per-request lines



async def warm_dns() -> None:
    """Resolve the USDA API host in the background as soon as the server starts.

    Its name goes through three CNAMEs (usda.gov -> api.data.gov -> cloud.gov) with
    60 s TTLs. On a freshly started Cloud Run instance, resolving it failed for up to
    ~40 s ("Temporary failure in name resolution") while other hosts resolved fine,
    so a user's first lookups fell back to estimates. Resolving it early gets that
    done before the first lookup needs it."""
    host = httpx.URL(SEARCH_URL).host
    loop = asyncio.get_running_loop()
    start = time.monotonic()
    while time.monotonic() - start < DNS_WARMUP_SECONDS:
        try:
            await loop.getaddrinfo(host, 443)
            log.info("DNS for %s ready after %.1f s", host, time.monotonic() - start)
            return
        except OSError as e:
            log.info("DNS for %s not ready yet (%s)", host, e)
            await asyncio.sleep(2)
    log.warning("DNS for %s still failing after %d s", host, DNS_WARMUP_SECONDS)


_client: httpx.AsyncClient | None = None


def usda_client() -> httpx.AsyncClient:
    """One client for the server's lifetime: it keeps the HTTPS connection to USDA
    open between lookups, so most lookups skip DNS and the TLS handshake (USDA's
    hostname is a three-CNAME chain whose resolution intermittently failed on
    Cloud Run)."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=TIMEOUT_SECONDS,
                                    limits=httpx.Limits(keepalive_expiry=120))
    return _client


@asynccontextmanager
async def lifespan(server: FastMCP):
    task = asyncio.create_task(warm_dns())
    try:
        yield {}
    finally:
        task.cancel()
        if _client is not None:
            await _client.aclose()


mcp = FastMCP("nutrition-lookup", lifespan=lifespan)


def api_key() -> str:
    key = os.environ.get("USDA_API_KEY", "")
    if key:
        return key
    log.warning("USDA_API_KEY not set; using DEMO_KEY (about 10 requests/hour)")
    return "DEMO_KEY"


def macros_per_100g(food: dict) -> dict | None:
    """Calories and macros per 100 g from one FDC search hit, or None if the food
    lacks the data."""
    by_number = {
        n.get("nutrientNumber"): n.get("value")
        for n in food.get("foodNutrients", [])
        if n.get("unitName", "").upper() in ("KCAL", "G")
    }
    macros = {name: by_number.get(number) for name, number in MACRO_NUMBERS.items()}
    if any(v is None for v in macros.values()):
        return None
    calories = next((by_number[n] for n in ENERGY_NUMBERS if by_number.get(n) is not None), None)
    if calories is None:
        calories = 4 * macros["protein_g"] + 4 * macros["carbs_g"] + 9 * macros["fat_g"]
    return {"calories": calories, **macros}


def scale(per_100g: dict, grams: float) -> dict:
    return {k: round(v * grams / 100, 1) for k, v in per_100g.items()}


def _main_name(description: str) -> str:
    """USDA names lead with the food itself: "Apples, raw, with skin" vs
    "Croissants, apple". The first word is what the food is."""
    return description.split(",")[0].split()[0].lower() if description.strip() else ""


def rank(foods: list[dict], query: str) -> list[dict]:
    """Stable re-order: foods whose main name matches a query word come first
    (prefix match covers plurals: apple/apples, tomato/tomatoes)."""
    words = [w for w in query.lower().split() if len(w) > 2]

    def is_main_match(food: dict) -> bool:
        name = _main_name(food.get("description", ""))
        return any(name.startswith(w) or w.startswith(name) for w in words if name)

    return sorted(foods, key=lambda f: not is_main_match(f))


def summarize(foods: list[dict], grams: float) -> list[dict]:
    matches = []
    for food in foods:
        per_100g = macros_per_100g(food)
        if per_100g is None:
            continue
        matches.append({
            "fdc_id": food["fdcId"],
            "description": food["description"],
            "per_100g": {k: round(v, 1) for k, v in per_100g.items()},
            "for_requested_grams": scale(per_100g, grams),
        })
        if len(matches) == MAX_MATCHES:
            break
    return matches


async def post_search(food_name: str) -> httpx.Response:
    """POST the USDA search, retrying when the connection can't be made or USDA says
    it's temporarily unavailable (502/503/504); in both cases the search never ran, so
    retrying is safe. Both were seen live: DNS for the API failing ("Temporary failure
    in name resolution") on Cloud Run, and a one-off 503."""
    client = usda_client()
    for attempt in range(ATTEMPTS):
        last = attempt == ATTEMPTS - 1
        try:
            # Key in a header, not the URL: request URLs end up in logs.
            response = await client.post(
                SEARCH_URL,
                headers={"X-Api-Key": api_key()},
                json={"query": food_name, "dataType": DATA_TYPES, "pageSize": 10},
            )
        except httpx.ConnectError as e:
            if last:
                raise
            log.warning("USDA connection failed (%s); retrying", e)
        else:
            if response.status_code not in RETRY_STATUSES or last:
                return response
            log.warning("USDA returned HTTP %s; retrying", response.status_code)
        await asyncio.sleep(1)


@mcp.tool()
async def lookup_nutrition(
    food_name: Annotated[str, Field(min_length=1, max_length=100,
                                    description="One food, base food first, then how it was prepared, "
                                                "e.g. 'apple raw', 'oats cooked', 'chicken breast roasted'.")],
    grams: Annotated[float, Field(gt=0, le=5000, description="Weight eaten, in grams.")],
) -> dict:
    """Look up calories and macros (USDA FoodData Central) for a single food at a given
    weight. Returns up to 5 candidate matches; check that the description fits."""
    try:
        return await search(food_name, grams)
    except RuntimeError as e:
        # The error goes back to the agent as the tool result; log it here too, or the
        # reason is lost from the server logs.
        log.warning("lookup_nutrition(%r, %sg) failed: %s", food_name, grams, e)
        raise


async def search(food_name: str, grams: float) -> dict:
    try:
        response = await post_search(food_name)
    except httpx.HTTPError as e:
        raise RuntimeError(f"USDA FoodData Central is unreachable: {type(e).__name__} {e}") from e
    if response.status_code == 429:
        raise RuntimeError("USDA FoodData Central rate limit reached; try again later.")
    if response.status_code != 200:
        raise RuntimeError(f"USDA FoodData Central returned HTTP {response.status_code}.")
    try:
        foods = response.json()["foods"]
    except (ValueError, KeyError) as e:
        raise RuntimeError("USDA FoodData Central sent a malformed response.") from e

    matches = summarize(rank(foods, food_name), grams)
    log.info("lookup_nutrition(%r, %sg) -> %s", food_name, grams,
             [m["description"] for m in matches])
    return {
        "food_name": food_name,
        "grams": grams,
        "source": "USDA FoodData Central",
        "matches": matches,
    }


if __name__ == "__main__":
    mcp.run()
