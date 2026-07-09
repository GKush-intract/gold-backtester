# tests/test_builder_page.py
import ast
from pathlib import Path

PAGE = Path(__file__).parent.parent / "pages" / "1_Strategy_Builder.py"


def test_page_exists_and_parses():
    assert PAGE.exists()
    ast.parse(PAGE.read_text())


def test_page_uses_shared_modules():
    src = PAGE.read_text()
    for needle in ["src.builder", "render_results", "render_param_inputs",
                   "generate_validated", "run_interview_turn", "run_revision_turn",
                   "Revision plan", "render_replay", "Diagnose with AI", "build_digest"]:
        assert needle in src, f"page should use {needle}"
