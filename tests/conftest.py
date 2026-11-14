import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retailpricing.core.config import load_config  # noqa: E402
from retailpricing.data_gen.generator import generate  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def small(cfg):
    df, truth = generate(cfg, seed=11, n_quotes=20_000)
    return df, truth
