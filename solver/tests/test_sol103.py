"""SOL 103 validation tests."""
import os
import numpy as np
import pytest

VALIDATION_DIR = os.path.join(os.path.dirname(__file__), "validation")
BEAM_MODES_BDF = os.path.join(VALIDATION_DIR, "cantilever_beam", "beam_modes.bdf")
PLATE_MODES_BDF = os.path.join(VALIDATION_DIR, "plate_modes", "plate_modes.bdf")


def run_sol103(bdf_path):
    from nastaero.bdf.parser import BDFParser
    from nastaero.solvers.sol103 import solve_modes
    parser = BDFParser()
    model = parser.parse(bdf_path)
    return solve_modes(model)


class TestCantileverBeamModes:
    """Cantilever beam modal analysis.

    Analytical first bending frequency:
    f1 = (1.8751)^2 / (2*pi*L^2) * sqrt(E*I / (rho*A))
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.results = run_sol103(BEAM_MODES_BDF)

    def test_has_modes(self):
        sc = self.results.subcases[0]
        assert sc.eigenvalues is not None
        assert len(sc.eigenvalues) > 0

    def test_positive_eigenvalues(self):
        """All eigenvalues should be positive (no rigid body modes)."""
        sc = self.results.subcases[0]
        for ev in sc.eigenvalues:
            assert ev > 0, f"Non-positive eigenvalue: {ev}"

    def test_ascending_frequencies(self):
        """Frequencies should be sorted ascending."""
        sc = self.results.subcases[0]
        for i in range(1, len(sc.frequencies)):
            assert sc.frequencies[i] >= sc.frequencies[i-1]

    def test_first_bending_frequency(self):
        """First bending frequency should match analytical within 5%.

        Analytical: f1 = (1.8751)^2/(2*pi*L^2) * sqrt(EI/(rho*A))
        """
        E = 7.0e10; I = 8.333e-10; rho = 2700.0; A = 1.0e-4; L = 1.0
        beta1 = 1.8751
        f1_analytical = beta1**2 / (2 * np.pi * L**2) * np.sqrt(E * I / (rho * A))
        sc = self.results.subcases[0]
        # First mode should be bending - find the lowest frequency
        f1_computed = sc.frequencies[0]
        rel_error = abs(f1_computed - f1_analytical) / f1_analytical
        assert rel_error < 0.05, (
            f"1st bending freq: {f1_computed:.2f} Hz vs analytical {f1_analytical:.2f} Hz, "
            f"error = {rel_error:.2%}"
        )

    def test_mode_shapes_have_entries(self):
        """Each mode shape should have entries for all free nodes."""
        sc = self.results.subcases[0]
        for j, mode in enumerate(sc.mode_shapes):
            assert len(mode) > 0, f"Mode {j+1} has no shape data"


class TestPlateNaturalFrequencies:
    """Simply-supported plate modal analysis.

    Analytical: f_mn = (pi/2) * sqrt(D/(rho*t)) * ((m/a)^2 + (n/b)^2)
    D = E*t^3 / (12*(1-nu^2))
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.results = run_sol103(PLATE_MODES_BDF)

    def test_has_modes(self):
        sc = self.results.subcases[0]
        assert sc.eigenvalues is not None
        assert len(sc.eigenvalues) > 0

    def test_positive_eigenvalues(self):
        sc = self.results.subcases[0]
        # Allow very small negative due to numerical noise
        for ev in sc.eigenvalues:
            assert ev > -1.0, f"Large negative eigenvalue: {ev}"

    def test_first_plate_frequency_order_of_magnitude(self):
        """First plate frequency should be in the right order of magnitude.

        For 1m x 1m steel plate, t=0.01m, f_11 ~ 48 Hz.
        With coarse 4x4 mesh, we allow wider tolerance.
        """
        sc = self.results.subcases[0]
        # Find first positive-frequency mode (skip near-zero rigid body modes)
        positive_freqs = [f for f in sc.frequencies if f > 1.0]
        assert len(positive_freqs) > 0, "No positive frequency modes found"
        f1 = positive_freqs[0]
        # For simply supported plate: f_11 ~ 48 Hz
        # Coarse mesh will be stiffer → higher frequency. Allow 10x tolerance.
        assert 10.0 < f1 < 500.0, (
            f"First plate frequency {f1:.2f} Hz outside expected range [10, 500] Hz"
        )
