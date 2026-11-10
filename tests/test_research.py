from recipes import Result, render_answer
from research import result_urls, strip_unknown_urls

SEARCH_OUTPUT = """Title: Added sugars | WHO
URL: https://www.who.int/news/sugars/
Snippet: WHO recommends reducing free sugars to less than 10% of total energy intake.

Title: Dietary Guidelines for Americans
URL: https://www.dietaryguidelines.gov/current
Snippet: Limit added sugars to less than 10 percent of calories per day."""

ALLOWED = result_urls([SEARCH_OUTPUT])


def test_result_urls_are_normalized():
    assert ALLOWED == {"https://www.who.int/news/sugars", "https://www.dietaryguidelines.gov/current"}


def test_links_from_the_search_are_kept():
    answer = ("WHO says under 10% of energy (https://www.who.int/news/sugars/). "
              "See also [the US guidelines](https://www.dietaryguidelines.gov/current).")
    assert strip_unknown_urls(answer, ALLOWED) == (answer, 0)


def test_invented_links_are_removed_but_markdown_text_stays():
    answer = ("Per [a Harvard study](https://madeup.edu/study), sugar is bad. "
              "More at https://another-madeup.com/page.")
    text, removed = strip_unknown_urls(answer, ALLOWED)
    assert removed == 2
    assert "madeup" not in text
    assert "Per a Harvard study, sugar is bad." in text


def test_recipe_turn_keeps_cited_search_links_and_drops_invented_ones():
    results = {"R1": Result("Light", "Chicken Bowl", "https://a.com/bowl", "Quick bowl.")}
    answer = ("[[R1]] — quick\n"
              "Sugar guidance: https://www.who.int/news/sugars/\n"
              "- **Fake Stew** — https://madeup.com/stew")
    rendered = render_answer(answer, results, ALLOWED)
    assert "https://a.com/bowl" in rendered
    assert "https://www.who.int/news/sugars/" in rendered
    assert "madeup.com" not in rendered


def test_bracket_citations_become_plain_links_and_real_urls_survive():
    # The format that previously made the filter drop real links.
    answer = ("WHO: under 10% of energy【https://www.who.int/news/sugars/†L3-L9】. "
              "Also limited in the US【2†source】.")
    text, removed = strip_unknown_urls(answer, ALLOWED)
    assert removed == 0
    assert "(https://www.who.int/news/sugars/)" in text
    assert "【" not in text and "†" not in text


def test_bracket_citation_with_an_invented_url_is_removed():
    text, removed = strip_unknown_urls("Sugar is bad【https://madeup.org/x†L1】.", ALLOWED)
    assert removed == 1 and "madeup" not in text and "【" not in text
    assert text == "Sugar is bad."  # no empty "()" left behind
