import sys
from pathlib import Path


# preprocess_captions.py is also a directly executable script and imports its sibling
# packages (`utils`, `captioners`) by their inner-repository names. Add that same inner
# root here so this test checks the prompt actually used by the CLI, not a copied string.
INNER_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INNER_REPO))

from remembr.scripts.preprocess_captions import CAPTION_STYLES, TERSE_QUERY


def test_terse_style_keeps_paper_reproduction_separate():
    assert CAPTION_STYLES["paper"] is None
    assert CAPTION_STYLES["terse"] == TERSE_QUERY


def test_terse_style_puts_searchable_evidence_first_without_guessing():
    assert "At most five short sentences" in TERSE_QUERY
    assert "Lead with the most identifiable object or landmark" in TERSE_QUERY
    assert "Add up to three others" in TERSE_QUERY
    assert "including a small or edge object with text, controls" in TERSE_QUERY
    assert "distinctive color, shape, or state" in TERSE_QUERY
    assert "broad visible type and distinguishing feature" in TERSE_QUERY
    assert "avoid repetition" not in TERSE_QUERY
    assert "use place names only when those words are readable" in TERSE_QUERY
    assert "If uncertain" in TERSE_QUERY
    assert "readable text without guessing" in TERSE_QUERY
    assert "Do not list other absent objects" in TERSE_QUERY
    assert "or say none are visible" in TERSE_QUERY
    assert "or say there are none" not in TERSE_QUERY
    assert "under 200 words" in TERSE_QUERY
    assert len(TERSE_QUERY.split()) <= 150


def test_terse_style_does_not_hardcode_evaluation_object_categories():
    prompt = TERSE_QUERY.lower()
    for category in ("seating", "waste", "recycling", "hydrant", "sculpture", "elevator", "rack"):
        assert category not in prompt
