from pathlib import Path

from src.strategies import get_strategy_registry

GENERATED_DIR = Path(__file__).parent.parent / "src" / "strategies" / "generated"

FIXTURE = '''\
from __future__ import annotations

from ...strategy import Strategy


class RegistryProbe(Strategy):
    name = "Registry Probe"
    params = {"n": ("int", 5, 1, 100, "test param")}

    def on_bar(self, ctx):
        pass
'''


def test_generated_dir_is_scanned(tmp_path):
    probe = GENERATED_DIR / "gen_registry_probe.py"
    probe.write_text(FIXTURE)
    try:
        registry = get_strategy_registry()
        assert "Registry Probe" in registry
        assert "MA Crossover" in registry  # existing strategies still found
    finally:
        probe.unlink()


def test_registry_without_generated_files():
    registry = get_strategy_registry()
    assert "MA Crossover" in registry
