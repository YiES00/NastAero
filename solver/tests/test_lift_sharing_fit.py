"""Unit tests for LiftSharingModel.fit_to_data and fit_residuals."""
from __future__ import annotations

import numpy as np
import pytest

from nastaero.loads_analysis.transition_loads import LiftSharingModel


class TestFitResiduals:
    """fit_residuals only evaluates; does not fit."""

    def test_evaluates_at_current_params(self):
        # Default model: V_50=28, k=0.18
        model = LiftSharingModel(V_50=28.0, k=0.18)
        V_data = np.array([0.0, 28.0, 50.0, 67.0])
        lambda_data = np.array([0.02, 0.50, 0.92, 0.99])
        result = model.fit_residuals(V_data, lambda_data)

        # At V=V_50, fit should give exactly 0.5
        assert result["lambda_fit"][1] == pytest.approx(0.5, abs=1e-9)
        # Document actual RMS for k=0.18 (NOT 0.018 as paper Table 5.1
        # incorrectly stated; correct value is ~0.0317)
        assert result["rms"] == pytest.approx(0.0317, rel=0.05)

    def test_does_not_modify_model(self):
        model = LiftSharingModel(V_50=28.0, k=0.18)
        original_V50 = model.V_50
        original_k = model.k
        V_data = np.array([0.0, 28.0, 50.0])
        lambda_data = np.array([0.02, 0.50, 0.92])
        _ = model.fit_residuals(V_data, lambda_data)
        # Model parameters should be unchanged
        assert model.V_50 == original_V50
        assert model.k == original_k


class TestFitToData:
    """fit_to_data performs actual nonlinear least-squares fit."""

    def test_fit_recovers_known_parameters(self):
        # Generate synthetic data from known logistic
        true_V50 = 30.0
        true_k = 0.20
        V_data = np.linspace(0, 67, 12)
        lambda_data = 1.0 / (1.0 + np.exp(-true_k * (V_data - true_V50)))

        # Fit with bad initial guess
        fitted = LiftSharingModel.fit_to_data(
            V_data, lambda_data, V_50_init=20.0, k_init=0.10
        )

        # Should recover near-exactly (synthetic data, no noise)
        assert fitted.V_50 == pytest.approx(true_V50, rel=1e-4)
        assert fitted.k == pytest.approx(true_k, rel=1e-4)

    def test_fit_to_NASA_LC_data(self):
        """Fit to the dissertation's 4 NASA L+C data points."""
        V_data = np.array([0.0, 28.0, 50.0, 67.0])
        lambda_data = np.array([0.02, 0.50, 0.92, 0.99])

        # Fit
        fitted = LiftSharingModel.fit_to_data(V_data, lambda_data)

        # Compute residual
        residuals = fitted.fit_residuals(V_data, lambda_data)
        # Optimal fit should give RMS smaller than the dissertation's
        # initial guess (V_50=28, k=0.18) RMS=0.0317
        assert residuals["rms"] < 0.04
        # Sanity check: fit gives ~symmetric logistic
        assert 25.0 < fitted.V_50 < 35.0
        assert 0.10 < fitted.k < 0.40

    def test_fit_residuals_decrease_after_fitting(self):
        """fit_to_data should produce equal-or-better RMS than initial guess."""
        V_data = np.array([0.0, 28.0, 50.0, 67.0])
        lambda_data = np.array([0.02, 0.50, 0.92, 0.99])

        # Initial-guess model
        guess_model = LiftSharingModel(V_50=28.0, k=0.18)
        guess_rms = guess_model.fit_residuals(V_data, lambda_data)["rms"]

        # Fitted model
        fitted = LiftSharingModel.fit_to_data(V_data, lambda_data)
        fitted_rms = fitted.fit_residuals(V_data, lambda_data)["rms"]

        # Fit should be at least as good as initial guess (likely better)
        assert fitted_rms <= guess_rms + 1e-6
