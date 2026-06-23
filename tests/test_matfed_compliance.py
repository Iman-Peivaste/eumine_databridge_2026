"""
MatFed API v1 compliance tests for LISTEuMINePredictor.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

MATFED_API = ROOT.parent / "hackathon_ref" / "matfed-api-template"
sys.path.insert(0, str(MATFED_API))

from pymatgen.core import Structure  # noqa: E402

from eumine_databridge.matfed.predictor import LISTEuMINePredictor  # noqa: E402

# Step 4B bundle when available; Step 4 ensemble layout works today
MODEL_PATH = os.environ.get(
    "MATFED_MODEL_PATH",
    str(ROOT / "models" / "ensemble"),
)
SAMPLE_CIFS = list((MATFED_API / "tests" / "sample_structures").glob("*.cif"))


@pytest.fixture(scope="module")
def predictor():
    p = LISTEuMINePredictor()
    p.load_model(MODEL_PATH)
    return p


@pytest.fixture(scope="module")
def sample_structures():
    structures = [
        Structure.from_file(str(cif_path)) for cif_path in sorted(SAMPLE_CIFS)
    ]
    assert len(structures) >= 3, (
        f"Need at least 3 sample CIFs, found {len(structures)}"
    )
    return structures


class TestMatFedInterface:
    def test_1_load_model_runs_without_error(self):
        p = LISTEuMINePredictor()
        p.load_model(MODEL_PATH)
        assert p._loaded is True

    def test_2_predict_returns_correct_length(
        self, predictor, sample_structures
    ):
        results = predictor.predict(sample_structures)
        assert len(results) == len(sample_structures)

    def test_3_predict_output_has_required_keys(
        self, predictor, sample_structures
    ):
        required_keys = {
            "formation_energy_per_atom",
            "band_gap",
            "model_id",
            "data_sources_used",
        }
        results = predictor.predict(sample_structures)
        for i, result in enumerate(results):
            missing = required_keys - set(result.keys())
            assert not missing, f"Structure {i}: missing keys {missing}"

    def test_4_predict_output_types_are_valid(
        self, predictor, sample_structures
    ):
        results = predictor.predict(sample_structures)
        for i, result in enumerate(results):
            assert isinstance(result["formation_energy_per_atom"], float)
            assert isinstance(result["band_gap"], float)
            assert isinstance(result["model_id"], str)
            assert isinstance(result["data_sources_used"], list)
            assert all(isinstance(s, str) for s in result["data_sources_used"])

    def test_5_predict_values_are_physically_reasonable(
        self, predictor, sample_structures
    ):
        results = predictor.predict(sample_structures)
        for i, result in enumerate(results):
            ef = result["formation_energy_per_atom"]
            bg = result["band_gap"]
            assert -6.0 <= ef <= 3.0, f"Structure {i}: EF={ef:.3f} out of range"
            assert 0.0 <= bg <= 15.0, f"Structure {i}: BG={bg:.3f} out of range"

    def test_6_describe_returns_required_fields(self, predictor):
        required_fields = {
            "team_name",
            "model_type",
            "api_version",
            "data_sources",
        }
        desc = predictor.describe()
        assert isinstance(desc, dict)
        missing = required_fields - set(desc.keys())
        assert not missing, f"describe() missing fields: {missing}"

    def test_7_describe_api_version_is_correct(self, predictor):
        desc = predictor.describe()
        assert desc["api_version"] == "1.0"

    def test_8_predict_single_structure(self, predictor, sample_structures):
        result = predictor.predict([sample_structures[0]])
        assert len(result) == 1
        assert "formation_energy_per_atom" in result[0]
        assert "band_gap" in result[0]

    def test_9_predict_empty_list(self, predictor):
        assert predictor.predict([]) == []

    def test_10_schema_validation_passes(self, predictor, sample_structures):
        from matfed_api import validate_predictions

        results = predictor.predict(sample_structures)
        validate_predictions(results)
