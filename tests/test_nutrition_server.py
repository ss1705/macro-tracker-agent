import asyncio

import httpx
import pytest

from tools import nutrition_server
from tools.nutrition_server import macros_per_100g, rank, scale, summarize

# Shaped like a real /foods/search hit (SR Legacy egg, values per 100 g).
EGG = {
    "fdcId": 171287,
    "description": "Egg, whole, raw, fresh",
    "foodNutrients": [
        {"nutrientNumber": "208", "value": 143, "unitName": "KCAL"},
        {"nutrientNumber": "268", "value": 599, "unitName": "kJ"},
        {"nutrientNumber": "203", "value": 12.6, "unitName": "G"},
        {"nutrientNumber": "204", "value": 9.51, "unitName": "G"},
        {"nutrientNumber": "205", "value": 0.72, "unitName": "G"},
    ],
}


def test_macros_per_100g_reads_nutrient_numbers():
    assert macros_per_100g(EGG) == {"calories": 143, "protein_g": 12.6, "fat_g": 9.51, "carbs_g": 0.72}


def test_atwater_energy_is_used_when_208_is_missing():
    food = {"foodNutrients": [n for n in EGG["foodNutrients"] if n["nutrientNumber"] != "208"]
            + [{"nutrientNumber": "958", "value": 140, "unitName": "KCAL"}]}
    assert macros_per_100g(food)["calories"] == 140


def test_calories_computed_from_macros_when_no_energy_reported():
    food = {"foodNutrients": [n for n in EGG["foodNutrients"] if n["nutrientNumber"] != "208"]}
    assert macros_per_100g(food)["calories"] == 4 * 12.6 + 4 * 0.72 + 9 * 9.51


def test_food_missing_a_macro_is_skipped():
    food = {"foodNutrients": [n for n in EGG["foodNutrients"] if n["nutrientNumber"] != "203"]}
    assert macros_per_100g(food) is None


def test_scale_to_grams():
    # two large eggs ~ 100 g
    assert scale({"calories": 143, "protein_g": 12.6}, 100) == {"calories": 143.0, "protein_g": 12.6}
    assert scale({"calories": 143, "protein_g": 12.6}, 50) == {"calories": 71.5, "protein_g": 6.3}


def test_summarize_caps_matches_and_skips_incomplete_foods():
    incomplete = {"fdcId": 1, "description": "x", "foodNutrients": []}
    matches = summarize([incomplete] + [EGG] * 7, grams=50)
    assert len(matches) == 5
    assert matches[0]["description"] == "Egg, whole, raw, fresh"
    assert matches[0]["for_requested_grams"]["protein_g"] == 6.3


def _named(*descriptions):
    return [{"description": d} for d in descriptions]


def test_rank_puts_the_food_itself_before_dishes_containing_it():
    # The real USDA order for "apple" led with these three.
    foods = _named("Croissants, apple", "Strudel, apple", "Babyfood, juice, apple",
                   "Apples, raw, with skin", "Apples, dried, sulfured")
    assert [f["description"] for f in rank(foods, "apple")][:2] == [
        "Apples, raw, with skin", "Apples, dried, sulfured"]


def test_rank_matches_any_query_word_and_keeps_order_otherwise():
    foods = _named("Cereals, oats, instant", "Rice, brown, cooked", "Rice, white, cooked")
    assert [f["description"] for f in rank(foods, "cooked brown rice")] == [
        "Rice, brown, cooked", "Rice, white, cooked", "Cereals, oats, instant"]
    eggs = _named("Egg substitute, liquid", "Egg, whole, cooked, hard-boiled")
    assert rank(eggs, "boiled egg") == eggs  # both are eggs: USDA's order kept


def fake_post(failures: int, failure: str = "connect"):
    """httpx.AsyncClient.post that fails `failures` times (can't connect, or HTTP 503),
    then answers."""
    calls = []

    async def post(self, url, **kwargs):
        calls.append(url)
        request = httpx.Request("POST", url)
        if len(calls) <= failures:
            if failure == "connect":
                raise httpx.ConnectError("Temporary failure in name resolution")
            return httpx.Response(503, request=request)
        return httpx.Response(200, json={"foods": [EGG]}, request=request)
    return post, calls


@pytest.fixture
def no_sleep(monkeypatch):
    async def instant(_):
        pass
    monkeypatch.setattr(nutrition_server.asyncio, "sleep", instant)
    monkeypatch.setattr(nutrition_server, "_client", None)  # fresh client per test loop


@pytest.mark.parametrize("failure", ["connect", "503"])
def test_lookup_retries_when_usda_is_briefly_unavailable_then_succeeds(monkeypatch, no_sleep, failure):
    post, calls = fake_post(failures=1, failure=failure)
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    result = asyncio.run(nutrition_server.lookup_nutrition("egg raw", 50))
    assert len(calls) == 2 and result["matches"][0]["fdc_id"] == 171287


def test_lookup_gives_up_after_two_connection_failures(monkeypatch, no_sleep):
    post, calls = fake_post(failures=5)
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    with pytest.raises(RuntimeError, match="unreachable"):
        asyncio.run(nutrition_server.lookup_nutrition("egg raw", 50))
    assert len(calls) == 2


def test_lookup_reports_a_lasting_503_after_two_attempts(monkeypatch, no_sleep):
    post, calls = fake_post(failures=5, failure="503")
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        asyncio.run(nutrition_server.lookup_nutrition("egg raw", 50))
    assert len(calls) == 2
