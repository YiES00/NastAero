"""SOL 146 - Dynamic Aeroelastic Response validation tests.

Phase A tests verify:
1. Without aero coupling, SOL 146 == SOL 112 (modal transient)
2. At V=0, aero coupling vanishes (same as SOL 112)
3. Impulse response peaks at natural frequency
4. Aero coupling shifts natural frequencies
"""
from __future__ import annotations
import os
import numpy as np
import pytest

VALIDATION_DIR = os.path.join(os.path.dirname(__file__), "validation")
BEAM_MODES_BDF = os.path.join(VALIDATION_DIR, "cantilever_beam", "beam_modes.bdf")


def _make_cantilever_model():
    """Parse the cantilever beam BDF for dynamic tests."""
    from nastaero.bdf.parser import BDFParser
    parser = BDFParser()
    model = parser.parse(BEAM_MODES_BDF)
    return model


def _impulse_force(tip_node: int = 11, dof_comp: int = 3,
                   amplitude: float = 100.0,
                   pulse_duration: float = 0.001):
    """Create an impulse force function at a node."""
    def force_func(t, dof_mgr):
        n_total = dof_mgr.total_dof
        F = np.zeros(n_total)
        if t <= pulse_duration:
            dof_idx = dof_mgr.get_dof(tip_node, dof_comp)
            if 0 <= dof_idx < n_total:
                F[dof_idx] = amplitude
        return F
    return force_func


def _sine_force(tip_node: int = 11, dof_comp: int = 3,
                amplitude: float = 10.0, freq_hz: float = 5.0):
    """Create a sinusoidal force function at a node."""
    def force_func(t, dof_mgr):
        n_total = dof_mgr.total_dof
        F = np.zeros(n_total)
        dof_idx = dof_mgr.get_dof(tip_node, dof_comp)
        if 0 <= dof_idx < n_total:
            F[dof_idx] = amplitude * np.sin(2.0 * np.pi * freq_hz * t)
        return F
    return force_func


class TestSol146MatchesSol112NoAero:
    """Without aero coupling, SOL 146 must equal SOL 112."""

    @pytest.fixture(autouse=True)
    def setup(self):
        model = _make_cantilever_model()
        t_array = np.linspace(0, 0.1, 201)
        force_func = _impulse_force()
        n_modes = 10

        from nastaero.solvers.sol112 import solve_modal_transient
        self.sol112 = solve_modal_transient(
            model, force_func, t_array,
            n_modes=n_modes, zeta=0.02,
            output_node_ids=[11], output_interval=1)

        # Re-parse because cross_reference mutates
        model2 = _make_cantilever_model()

        from nastaero.solvers.sol146 import solve_aeroelastic_transient
        self.sol146 = solve_aeroelastic_transient(
            model2, force_func=force_func, t_array=t_array,
            n_modes=n_modes, zeta=0.02,
            use_aero_coupling=False,
            output_node_ids=[11], output_interval=1)

    def test_same_time_array(self):
        np.testing.assert_array_almost_equal(
            self.sol112["t"], self.sol146["t"], decimal=10)

    def test_same_frequencies(self):
        np.testing.assert_array_almost_equal(
            self.sol112["frequencies"], self.sol146["frequencies"],
            decimal=6)

    def test_same_tip_displacement(self):
        """Tip displacements should match within tight tolerance."""
        d112 = self.sol112["displacements"][11]
        d146 = self.sol146["displacements"][11]
        # Z-displacement (DOF 3, index 2)
        max_diff = np.max(np.abs(d112[:, 2] - d146[:, 2]))
        max_val = np.max(np.abs(d112[:, 2]))
        assert max_diff < 1e-10 * max(max_val, 1.0), (
            f"Tip Z displacement mismatch: max diff = {max_diff:.2e}, "
            f"max val = {max_val:.2e}")

    def test_physical_displacement_equivalence(self):
        """Physical displacements (Phi @ q) must match exactly.

        Modal coordinates may differ due to eigenvector sign ambiguity
        between separate eigsh calls, but the physical result u = Phi @ q
        is invariant.  The test_same_tip_displacement already verifies this.
        Here we verify the overall modal coord energy is consistent.
        """
        mc112 = self.sol112["modal_coords"]
        mc146 = self.sol146["modal_coords"]
        # Modal energy (sum of squares) should match regardless of sign
        energy_112 = np.sum(mc112 ** 2, axis=1)
        energy_146 = np.sum(mc146 ** 2, axis=1)
        max_diff = np.max(np.abs(energy_112 - energy_146))
        max_val = np.max(energy_112)
        assert max_diff < 1e-12 * max(max_val, 1.0), (
            f"Modal energy mismatch: max diff = {max_diff:.2e}")


class TestSol146ZeroVelocity:
    """At V=0, aero coupling vanishes -> same as SOL 112."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from nastaero.bdf.parser import BDFParser
        from nastaero.bdf.model import BDFModel
        from nastaero.bdf.cards.aero import AERO

        model = _make_cantilever_model()
        # Add a zero-velocity AERO card (no actual panels)
        model.aero = AERO(velocity=0.0, rhoref=1.225, refc=1.0)
        # No CAERO1 panels -> aero coupling should be skipped

        t_array = np.linspace(0, 0.1, 201)
        force_func = _impulse_force()
        n_modes = 10

        from nastaero.solvers.sol146 import solve_aeroelastic_transient
        self.result = solve_aeroelastic_transient(
            model, force_func=force_func, t_array=t_array,
            n_modes=n_modes, zeta=0.02,
            use_aero_coupling=True,
            output_node_ids=[11], output_interval=1)

    def test_aero_not_active(self):
        assert self.result["aero_coupling_active"] is False

    def test_has_displacements(self):
        d = self.result["displacements"][11]
        assert d.shape[0] > 0
        # Should have nonzero response from impulse
        assert np.max(np.abs(d[:, 2])) > 0


class TestSol146ImpulseResponse:
    """Unit impulse -> verify response is oscillatory at natural frequency."""

    @pytest.fixture(autouse=True)
    def setup(self):
        model = _make_cantilever_model()
        t_array = np.linspace(0, 0.5, 2001)
        force_func = _impulse_force(amplitude=100.0, pulse_duration=0.001)

        from nastaero.solvers.sol146 import solve_aeroelastic_transient
        self.result = solve_aeroelastic_transient(
            model, force_func=force_func, t_array=t_array,
            n_modes=10, zeta=0.005,
            use_aero_coupling=False,
            output_node_ids=[11], output_interval=1)

    def test_response_oscillates(self):
        """Response should oscillate (have zero crossings)."""
        d = self.result["displacements"][11][:, 2]
        # Skip first few points (transient)
        d_trim = d[10:]
        sign_changes = np.sum(np.diff(np.sign(d_trim)) != 0)
        assert sign_changes >= 4, (
            f"Expected oscillatory response, got {sign_changes} zero crossings")

    def test_peak_near_natural_frequency(self):
        """FFT of response should peak near first natural frequency."""
        d = self.result["displacements"][11][:, 2]
        t = self.result["t"]
        dt = t[1] - t[0]

        spectrum = np.abs(np.fft.rfft(d))
        freqs = np.fft.rfftfreq(len(d), d=dt)

        # Find peak frequency (skip DC bin)
        peak_idx = np.argmax(spectrum[1:]) + 1
        peak_freq = freqs[peak_idx]

        # Natural frequency of first bending mode
        f1_nat = self.result["frequencies"][0]

        # Peak should be within 20% of natural frequency
        rel_diff = abs(peak_freq - f1_nat) / f1_nat
        assert rel_diff < 0.20, (
            f"Peak freq {peak_freq:.2f} Hz vs natural {f1_nat:.2f} Hz, "
            f"rel diff = {rel_diff:.2%}")

    def test_response_decays(self):
        """With damping, response amplitude should decrease over time."""
        d = self.result["displacements"][11][:, 2]
        n = len(d)
        # Compare first quarter amplitude to last quarter
        first_quarter = np.max(np.abs(d[:n // 4]))
        last_quarter = np.max(np.abs(d[3 * n // 4:]))
        assert last_quarter < first_quarter, (
            f"Response should decay: first quarter max = {first_quarter:.4e}, "
            f"last quarter max = {last_quarter:.4e}")


class TestSol146WithAeroStiffness:
    """Aero coupling shifts natural frequencies (modifies stiffness)."""

    def test_aero_stiffness_modifies_response(self):
        """With aero stiffness, response should differ from no-aero case."""
        from nastaero.solvers.sol146 import solve_aeroelastic_transient
        from nastaero.bdf.cards.aero import AERO, CAERO1, PAERO1, SPLINE1
        from nastaero.bdf.cards.sets import SET1

        t_array = np.linspace(0, 0.1, 501)
        force_func = _impulse_force()
        n_modes = 10

        # Without aero
        model1 = _make_cantilever_model()
        res_no_aero = solve_aeroelastic_transient(
            model1, force_func=force_func, t_array=t_array,
            n_modes=n_modes, zeta=0.02, use_aero_coupling=False,
            output_node_ids=[11], output_interval=1)

        # With aero: add AERO card + CAERO1 panel + spline
        model2 = _make_cantilever_model()
        model2.aero = AERO(velocity=50.0, rhoref=1.225, refc=0.1)

        # Create a simple CAERO1 panel along the beam
        caero = CAERO1()
        caero.eid = 1001
        caero.pid = 100
        caero.nspan = 4
        caero.nchord = 1
        caero.p1 = np.array([0.0, -0.05, 0.0])
        caero.chord1 = 0.1
        caero.p4 = np.array([0.0, 0.95, 0.0])
        caero.chord4 = 0.1
        model2.caero_panels[caero.eid] = caero

        paero = PAERO1()
        paero.pid = 100
        model2.properties[paero.pid] = paero

        # Create SET1 with structural nodes
        set1 = SET1()
        set1.sid = 10
        set1.ids = list(range(1, 12))  # nodes 1-11
        model2.sets[set1.sid] = set1

        # Create SPLINE1 to connect aero to structure
        spline = SPLINE1()
        spline.eid = 200
        spline.caero = 1001
        spline.box1 = 1001
        spline.box2 = 1004
        spline.setg = 10
        model2.splines[spline.eid] = spline

        res_with_aero = solve_aeroelastic_transient(
            model2, force_func=force_func, t_array=t_array,
            n_modes=n_modes, zeta=0.02, use_aero_coupling=True,
            output_node_ids=[11], output_interval=1)

        # Aero coupling should be active
        assert res_with_aero["aero_coupling_active"] is True

        # Responses should differ (aero stiffness modifies dynamics)
        d_no_aero = res_no_aero["displacements"][11][:, 2]
        d_aero = res_with_aero["displacements"][11][:, 2]

        # They start the same (t=0) but diverge
        max_diff = np.max(np.abs(d_no_aero - d_aero))
        assert max_diff > 1e-12, (
            f"Aero coupling should change response, max_diff = {max_diff:.2e}")


class TestDynamicCardParsing:
    """Test BDF card parsers for dynamic analysis cards."""

    def test_tload1_parse(self):
        from nastaero.bdf.cards.dynamic import TLOAD1
        fields = ["TLOAD1", "10", "20", "0.0", "0", "30"]
        t = TLOAD1.from_fields(fields)
        assert t.sid == 10
        assert t.exciteid == 20
        assert t.tid == 30
        assert t.load_type == 0

    def test_dload_parse(self):
        from nastaero.bdf.cards.dynamic import DLOAD
        fields = ["DLOAD", "5", "2.0", "1.5", "10", "0.5", "20"]
        d = DLOAD.from_fields(fields)
        assert d.sid == 5
        assert d.scale == 2.0
        assert len(d.scale_factors) == 2
        assert d.scale_factors[0] == pytest.approx(1.5)
        assert d.load_ids == [10, 20]

    def test_tabled1_parse(self):
        from nastaero.bdf.cards.dynamic import TABLED1
        fields = ["TABLED1", "1", "", "",
                  "", "0.0", "0.0", "0.5", "1.0",
                  "1.0", "1.0", "1.5", "0.0", "ENDT"]
        t = TABLED1.from_fields(fields)
        assert t.tid == 1
        assert len(t.x) == 4
        np.testing.assert_array_almost_equal(t.x, [0.0, 0.5, 1.0, 1.5])
        np.testing.assert_array_almost_equal(t.y, [0.0, 1.0, 1.0, 0.0])

    def test_tabled1_interpolation(self):
        from nastaero.bdf.cards.dynamic import TABLED1
        t = TABLED1(tid=1,
                    x=np.array([0.0, 1.0, 2.0]),
                    y=np.array([0.0, 1.0, 0.0]))
        assert t.evaluate(0.0) == pytest.approx(0.0)
        assert t.evaluate(0.5) == pytest.approx(0.5)
        assert t.evaluate(1.0) == pytest.approx(1.0)
        assert t.evaluate(1.5) == pytest.approx(0.5)
        assert t.evaluate(2.0) == pytest.approx(0.0)

    def test_gust_parse(self):
        from nastaero.bdf.cards.dynamic import GUST
        fields = ["GUST", "1", "10", "5.0", "100.0", "200.0"]
        g = GUST.from_fields(fields)
        assert g.sid == 1
        assert g.dload_id == 10
        assert g.wg == pytest.approx(5.0)
        assert g.x0 == pytest.approx(100.0)
        assert g.v == pytest.approx(200.0)

    def test_darea_parse(self):
        from nastaero.bdf.cards.dynamic import DAREA
        fields = ["DAREA", "1", "100", "3", "1.5", "200", "3", "2.0"]
        d = DAREA.from_fields(fields)
        assert d.sid == 1
        assert len(d.entries) == 2
        assert d.entries[0] == (100, 3, pytest.approx(1.5))
        assert d.entries[1] == (200, 3, pytest.approx(2.0))

    def test_tstep_parse(self):
        from nastaero.bdf.cards.dynamic import TSTEP
        fields = ["TSTEP", "1", "1000", "0.001", "5"]
        t = TSTEP.from_fields(fields)
        assert t.sid == 1
        assert t.n_steps == 1000
        assert t.dt == pytest.approx(0.001)
        assert t.skip == 5

    def test_freq1_parse(self):
        from nastaero.bdf.cards.dynamic import FREQ1
        fields = ["FREQ1", "1", "0.5", "0.1", "100"]
        f = FREQ1.from_fields(fields)
        assert f.sid == 1
        assert f.f1 == pytest.approx(0.5)
        assert f.df == pytest.approx(0.1)
        assert f.ndf == 100

    def test_tabdmp1_parse(self):
        from nastaero.bdf.cards.dynamic import TABDMP1
        fields = ["TABDMP1", "1", "CRIT",
                  "1.0", "0.02", "10.0", "0.05", "ENDT"]
        t = TABDMP1.from_fields(fields)
        assert t.tid == 1
        assert t.damp_type == "CRIT"
        assert len(t.freqs) == 2
        np.testing.assert_array_almost_equal(t.freqs, [1.0, 10.0])
        np.testing.assert_array_almost_equal(t.values, [0.02, 0.05])

    def test_tabdmp1_interpolation(self):
        from nastaero.bdf.cards.dynamic import TABDMP1
        t = TABDMP1(tid=1, damp_type="CRIT",
                    freqs=np.array([1.0, 10.0]),
                    values=np.array([0.02, 0.05]))
        assert t.get_damping(1.0) == pytest.approx(0.02)
        assert t.get_damping(10.0) == pytest.approx(0.05)
        assert t.get_damping(5.5) == pytest.approx(0.035)
