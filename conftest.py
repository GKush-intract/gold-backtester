import sys
from pathlib import Path

# Make repo root importable so `import src...` works in tests.
sys.path.insert(0, str(Path(__file__).parent))


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: end-to-end browser smoke tests (need running server)")
