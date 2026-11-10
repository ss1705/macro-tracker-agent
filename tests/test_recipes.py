from recipes import (
    Result, Tier, clean_snippet, compute_tiers, keep_recipe_pages, parse_labelled_results,
    parse_results, render_answer,
)


def test_tiers_scale_from_the_meal_ceiling_when_lots_is_left():
    # 1,520 kcal / 136 g left: tiers come from the 1,000 kcal meal ceiling,
    # protein scales with each tier's share of the remaining calories.
    assert compute_tiers(1520, 136) == [
        Tier("Light", 400, 35),
        Tier("Medium", 700, 65),
        Tier("Hearty", 1000, 90),
    ]


def test_tiers_use_the_remaining_budget_when_it_is_below_the_ceiling():
    assert compute_tiers(600, 40) == [
        Tier("Light", 250, 15),
        Tier("Medium", 400, 25),
        Tier("Hearty", 600, 40),
    ]


def test_small_budget_drops_tiers_below_the_minimum():
    assert compute_tiers(300, 20) == [Tier("Medium", 200, 15), Tier("Hearty", 300, 20)]


def test_no_tiers_when_almost_nothing_or_less_is_left():
    assert compute_tiers(100, 10) == []
    assert compute_tiers(-250, -5) == []


def test_snack_tiers_use_the_smaller_ceiling():
    assert compute_tiers(2000, 150, "snack") == [
        Tier("Light", 150, 10),
        Tier("Medium", 300, 20),
        Tier("Hearty", 400, 30),
    ]


def test_protein_target_is_zero_when_protein_goal_is_already_met():
    assert all(t.min_protein_g == 0 for t in compute_tiers(800, -10))


def test_parse_results_reads_tavily_blocks():
    text = (
        "Detailed Results:\n\n"
        "Title: Salmon Bowl\nURL: https://a.com/salmon\nContent: Easy salmon bowl.\nMore text.\n\n"
        "Title: Tuna Wrap\nURL: https://b.com/tuna\nContent: Quick wrap."
    )
    assert parse_results(text) == [
        ("Salmon Bowl", "https://a.com/salmon", "Easy salmon bowl.\nMore text."),
        ("Tuna Wrap", "https://b.com/tuna", "Quick wrap."),
    ]


TOOL_OUTPUT = """## Light tier (searched: under 400 kcal, at least 35 g protein)
[R1] Title: Air Fryer Teriyaki Chicken
URL: https://a.com/teriyaki
Snippet: Sweet-savory chicken thighs, ready in 20 minutes.
[R2] Title: 15 High-Protein Meals Under 500 Calories
URL: https://b.com/meals
Snippet: Turkey taco bowls, shrimp stir-fry and more.

## Medium tier (searched: under 700 kcal, at least 60 g protein)
[R3] Title: Chicken Pizzaiola
URL: https://c.com/pizzaiola
Snippet: Chicken in a tomato and oregano sauce."""

RESULTS = parse_labelled_results([TOOL_OUTPUT])


def test_parse_labelled_results_keeps_tier_and_fields():
    assert RESULTS["R1"] == Result("Light", "Air Fryer Teriyaki Chicken", "https://a.com/teriyaki",
                                   "Sweet-savory chicken thighs, ready in 20 minutes.")
    assert RESULTS["R3"].tier == "Medium"


def test_picks_are_replaced_with_title_url_snippet_and_reason():
    answer = "Some ideas:\n**Light**\n[[R1]] — quick and high in protein\n**Medium**\n- [[R3]]: classic"
    rendered = render_answer(answer, RESULTS)
    assert "- **Air Fryer Teriyaki Chicken** — https://a.com/teriyaki" in rendered
    assert "Sweet-savory chicken thighs" in rendered
    assert "Why: quick and high in protein" in rendered
    assert "**Chicken Pizzaiola** — https://c.com/pizzaiola" in rendered
    assert "[[" not in rendered


def test_unknown_ids_and_model_typed_urls_are_dropped():
    answer = ("[[R1]] — good\n[[R9]] — invented\n"
              "- **Made Up Stew** — https://madeup.com/stew\nEnjoy!")
    rendered = render_answer(answer, RESULTS)
    assert "teriyaki" in rendered
    assert "invented" not in rendered and "madeup.com" not in rendered
    assert "Enjoy!" in rendered


def test_inline_reference_becomes_title_and_link():
    rendered = render_answer("If you're short on time, try [[R1]] tonight.", RESULTS)
    assert rendered == "If you're short on time, try **Air Fryer Teriyaki Chicken** (https://a.com/teriyaki) tonight."


def test_no_valid_picks_falls_back_to_top_result_per_tier():
    rendered = render_answer("Try a chickpea curry — https://madeup.com/curry", RESULTS)
    assert "madeup" not in rendered
    assert "Air Fryer Teriyaki Chicken" in rendered and "Chicken Pizzaiola" in rendered
    assert "15 High-Protein Meals" not in rendered  # only the top result per tier


def test_reply_that_picks_nothing_is_kept_without_adding_results():
    reply = "None of these fit a shrimp-free salad; try chicken or chickpeas instead."
    assert render_answer(reply, RESULTS) == reply


def test_answer_is_kept_when_search_found_nothing():
    assert render_answer("No recipes turned up, want to try another search?", {}) == \
        "No recipes turned up, want to try another search?"


def test_clean_snippet_strips_markdown_noise_and_cuts_at_a_word():
    snippet = clean_snippet("### Seitan Bowl [...] **46g** protein | " + "word " * 100)
    assert snippet.startswith("Seitan Bowl 46g protein word")
    assert snippet.endswith("…") and len(snippet) <= 221


def test_keep_recipe_pages_drops_plans_and_pdfs_and_puts_single_recipes_first():
    # Titles taken from real Tavily results during the recipe-quality comparison.
    results = [(title, url, "") for title, url in [
        ("20 Healthy Meals Under 400 Calories", "https://www.healthline.com/x"),
        ("Meal plan with over 80g protein w/ PDF - Strongr Fastr", "https://www.strongrfastr.com/80g"),
        ("High Protein / Low Calorie Lazy Chicken", "https://loseweighteatpizza.com/lazy-chicken"),
        ("High-Protein Low-Calorie Dessert Recipes | PDF | Chocolate", "https://www.scribd.com/document/1"),
        ("1000 Calorie Vegetarian Meal Plan", "https://www.eatthismuch.com/diet-plan/1000"),
        ("81 High-Protein Dinner Recipes | Good Food", "https://www.bbcgoodfood.com/recipes/collection/x"),
        ("7-Day High Protein Diet Meal Plan", "https://pescetarian.kitchen/plan"),
        ("Breakfast Quesadilla (High-Protein) - Skinnytaste", "https://www.skinnytaste.com/quesadilla/"),
    ]]
    assert [r[0] for r in keep_recipe_pages(results)] == [
        "High Protein / Low Calorie Lazy Chicken",
        "Breakfast Quesadilla (High-Protein) - Skinnytaste",
        "20 Healthy Meals Under 400 Calories",
        "81 High-Protein Dinner Recipes | Good Food",
    ]


def test_keep_recipe_pages_keeps_roundups_when_nothing_else_is_left():
    roundups = [("25 High-Protein Snacks", "https://a.com/1", ""), ("30 Easy Dinners", "https://b.com/2", "")]
    assert keep_recipe_pages(roundups) == roundups
