from src.tools.prompt_interpreter import interpret_prompt


def test_basic_extraction() -> None:
    plan = interpret_prompt("Please annotate all cars and people in the images.")
    assert "car" in plan.classes
    assert "person" in plan.classes


def test_synonyms() -> None:
    plan = interpret_prompt("find pedestrians and vehicles")
    assert plan.classes == ["person", "car"]


def test_bigram_traffic_light() -> None:
    plan = interpret_prompt("detect traffic light and stop sign")
    assert "traffic_light" in plan.classes
    assert "stop_sign" in plan.classes


def test_empty_falls_back() -> None:
    # Nothing meaningful in prompt (only stopwords) → fallback
    plan = interpret_prompt("the of and in or please", fallback_schema=["person"])
    assert plan.classes == ["person"]


def test_unknown_tokens_used_as_raw_classes() -> None:
    plan = interpret_prompt("widgets and gadgets", fallback_schema=["person"])
    assert "widget" in plan.classes
    assert "gadget" in plan.classes


def test_per_class_prompt_built() -> None:
    plan = interpret_prompt("cars and dogs")
    # SAM3 prefers single-word prompts
    assert plan.per_class_prompt.get("car") == "car"
    assert plan.per_class_prompt.get("dog") == "dog"


def test_modifier_backtracking_and_typos() -> None:
    plan = interpret_prompt("there is a metal \"box\" in silver color so anootate it")
    assert plan.classes == ["box"]
    assert plan.per_class_prompt["box"] == "metal silver color box"

    plan2 = interpret_prompt("metal box in silver color")
    assert plan2.classes == ["box"]
    assert plan2.per_class_prompt["box"] == "metal silver color box"


def test_custom_singularization() -> None:
    plan = interpret_prompt("find widgets and boxes")
    assert "widget" in plan.classes
    assert "box" in plan.classes


def test_two_known_classes_in_one_phrase_without_separator() -> None:
    """"near" isn't a phrase splitter, so both nouns land in one phrase's
    token list — the second (earlier) known class must not be silently
    absorbed into the first's prompt text and lost."""
    plan = interpret_prompt("find the big red truck near the small blue car")
    assert set(plan.classes) == {"truck", "car"}
    assert plan.per_class_prompt["truck"] == "truck"
    assert any("truck" in n and "car" in n for n in plan.notes)


def test_unrecognized_second_noun_still_not_rescued() -> None:
    """The rescue only fires for tokens that resolve to a *known* synonym-table
    class; a custom domain noun neither side recognizes stays a limitation of
    the rule-based matcher, not something this fix claims to solve."""
    plan = interpret_prompt("find scratches near dark spots")
    assert plan.classes == ["spot"]
