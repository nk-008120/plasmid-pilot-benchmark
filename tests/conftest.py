"""Shared fixtures. The synthetic builders live in pilot.synthetic so the smoke-test script can use them too."""
from pathlib import Path

import pytest

from pilot.synthetic import build_plasmid, random_dna, write_genbank  # noqa: F401  (re-exported for the tests)


@pytest.fixture
def make_gb(tmp_path):
    def _make(name, seed, specs, rotate_to_wrap=False, topology="circular", directory=None):
        seq, planted = build_plasmid(seed, specs, rotate_to_wrap)
        folder = directory or tmp_path
        path = Path(folder) / f"{name}.gb"
        write_genbank(path, seq, planted, name=name[:16], topology=topology)
        return path, seq, planted

    return _make
