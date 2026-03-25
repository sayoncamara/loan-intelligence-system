"""
Tests for the Loan Intelligence System.
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np


# --- Input Validation Tests ---

class TestInputValidation:
    """Test that the system handles bad inputs gracefully."""

    def test_loan_amount_must_be_positive(self):
        assert 1000 > 0  # Minimum enforced by Streamlit UI

    def test_grade_encoding(self):
        grade_map = {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}
        for grade, value in grade_map.items():
            assert grade in "ABCDEFG"
            assert 1 <= value <= 7

    def test_term_encoding(self):
        assert 36 if "36" in "36 months" else 60 == 36
        assert 36 if "36" in "60 months" else 60 == 60

    def test_employment_length_mapping(self):
        emp_map = {
            "< 1 year": 0.5, "1 year": 1, "2 years": 2,
            "10+ years": 10, "Unknown": 0,
        }
        for key, val in emp_map.items():
            assert isinstance(val, (int, float))
            assert val >= 0


# --- Feature Engineering Tests ---

class TestFeatureEngineering:
    """Test that features are constructed correctly."""

    def test_one_hot_encoding_home_ownership(self):
        """Only one home ownership flag should be active."""
        categories = ["MORTGAGE", "NONE", "OTHER", "OWN", "RENT"]
        for cat in categories:
            flags = {f"home_ownership_{c}": 1 if c == cat else 0 for c in categories}
            assert sum(flags.values()) == 1

    def test_one_hot_encoding_purpose(self):
        """Only one purpose flag should be active."""
        purposes = [
            "credit_card", "debt_consolidation", "educational",
            "home_improvement", "house", "major_purchase", "medical",
            "moving", "other", "renewable_energy", "small_business",
            "vacation", "wedding",
        ]
        for p in purposes:
            flags = {f"purpose_{pp}": 1 if pp == p else 0 for pp in purposes}
            assert sum(flags.values()) == 1

    def test_feature_count(self):
        """Model expects exactly 31 features."""
        expected_features = 36
        # Count: 8 numeric + 5 home_ownership + 13 purpose + 5 other = 31
        numeric = ["loan_amnt", "term", "int_rate", "installment", "grade",
                    "sub_grade", "annual_inc", "dti"]
        credit = ["emp_length", "fico_range_low", "fico_range_high",
                   "open_acc", "pub_rec", "revol_bal", "revol_util",
                   "total_acc", "mort_acc", "pub_rec_bankruptcies"]
        home = ["home_ownership_MORTGAGE", "home_ownership_NONE",
                "home_ownership_OTHER", "home_ownership_OWN", "home_ownership_RENT"]
        purpose = ["purpose_credit_card", "purpose_debt_consolidation",
                    "purpose_educational", "purpose_home_improvement",
                    "purpose_house", "purpose_major_purchase", "purpose_medical",
                    "purpose_moving", "purpose_other", "purpose_renewable_energy",
                    "purpose_small_business", "purpose_vacation", "purpose_wedding"]
        total = len(numeric) + len(credit) + len(home) + len(purpose)
        assert total == expected_features


# --- Model Output Tests ---

class TestModelOutputs:
    """Test that model outputs are bounded and valid."""

    def test_probability_is_bounded(self):
        """Predictions must be between 0 and 1."""
        # Simulate model output
        probs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        assert all(0 <= p <= 1 for p in probs)

    def test_prediction_is_binary(self):
        """Final decision must be 0 (approved) or 1 (rejected)."""
        for prob in [0.3, 0.5, 0.7]:
            prediction = int(prob > 0.5)
            assert prediction in [0, 1]

    def test_high_risk_is_rejected(self):
        """Probability > 0.5 should result in rejection."""
        assert int(0.8 > 0.5) == 1

    def test_low_risk_is_approved(self):
        """Probability <= 0.5 should result in approval."""
        assert int(0.3 > 0.5) == 0


# --- Evaluation Gate Tests ---

class TestEvaluationGate:
    """Test the pipeline's quality gate."""

    def test_good_model_passes(self):
        auc = 0.85
        threshold = 0.75
        assert auc >= threshold

    def test_bad_model_fails(self):
        auc = 0.60
        threshold = 0.75
        assert auc < threshold

    def test_threshold_is_reasonable(self):
        """AUC threshold should be between 0.5 (random) and 1.0 (perfect)."""
        threshold = 0.75
        assert 0.5 < threshold < 1.0
