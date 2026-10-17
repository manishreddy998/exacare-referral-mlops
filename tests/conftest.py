import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def packet_dir(tmp_path_factory):
    from referral_mlops.synth.build import build_batch
    d = tmp_path_factory.mktemp("packets")
    build_batch(d, 6, seed=11, prefix="TST")
    return d


@pytest.fixture
def rng():
    return random.Random(0)
