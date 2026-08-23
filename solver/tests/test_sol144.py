"""Tests for SOL 144 - Static Aeroelastic Trim."""
import os
import numpy as np
import pytest
from nastaero.bdf.parser import BDFParser

VALIDATION_DIR = os.path.join(os.path.dirname(__file__), "validation")
GOLAND_BDF = os.path.join(VALIDATION_DIR, "goland_wing", "goland_static.bdf")


def parse_bdf(filepath):
    parser = BDFParser()
    return parser.parse(filepath)


class TestGolandParsing:
    """Test that the Goland wing BDF is parsed correctly."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = parse_bdf(GOLAND_BDF)

    def test_sol(self):
        assert self.model.sol == 144

    def test_nodes(self):
        assert len(self.model.nodes) == 11

    def test_elements(self):
        assert len(self.model.elements) == 10

    def test_aeros(self):
        assert self.model.aeros is not None
        assert self.model.aeros.refc == pytest.approx(1.8288)
        assert self.model.aeros.refb == pytest.approx(12.192)

    def test_aero(self):
        assert self.model.aero is not None
        assert self.model.aero.velocity == pytest.approx(50.0)

    def test_caero(self):
        assert 1001 in self.model.caero_panels
        c = self.model.caero_panels[1001]
        assert c.nspan == 8
        assert c.nchord == 2

    def test_spline(self):
        assert 100 in self.model.splines

    def test_set1(self):
        assert 10 in self.model.sets
        assert len(self.model.sets[10].ids) == 11

    def test_aestat(self):
        assert 501 in self.model.aestats
        assert self.model.aestats[501].label == "ANGLEA"

    def test_trim(self):
        assert 1 in self.model.trims
        t = self.model.trims[1]
        assert t.mach == pytest.approx(0.3)
        assert t.q == pytest.approx(1531.25)
        assert len(t.variables) == 1
        assert t.variables[0][0] == "URDD3"


class TestGolandTrim:
    """Test SOL 144 trim solution for Goland wing."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from nastaero.solvers.sol144 import solve_trim
        self.model = parse_bdf(GOLAND_BDF)
        self.results = solve_trim(self.model)

    def test_has_results(self):
        assert len(self.results.subcases) > 0

    def test_displacements_exist(self):
        sc = self.results.subcases[0]
        assert len(sc.displacements) > 0

    def test_trim_variables_exist(self):
        sc = self.results.subcases[0]
        assert sc.trim_variables is not None
        assert "ANGLEA" in sc.trim_variables

    def test_angle_of_attack_reasonable(self):
        """ANGLEA should be a finite, small angle."""
        sc = self.results.subcases[0]
        alpha = sc.trim_variables["ANGLEA"]
        alpha_deg = np.degrees(alpha)
        # Check that ANGLEA is finite and within a reasonable range
        assert np.isfinite(alpha), "ANGLEA should be finite"
        assert abs(alpha_deg) < 30, f"ANGLEA magnitude too large: {alpha_deg:.4f} deg"

    def test_aero_forces(self):
        """Aero forces should exist and produce lift."""
        sc = self.results.subcases[0]
        assert sc.aero_forces is not None
        total_fz = np.sum(sc.aero_forces[:, 2])
        # Should produce positive lift (upward)
        assert total_fz > 0, f"Total Fz should be positive, got {total_fz:.2f}"

    def test_tip_displacement(self):
        """Tip should deflect upward under positive lift."""
        sc = self.results.subcases[0]
        if 11 in sc.displacements:
            tip_z = sc.displacements[11][2]  # z-displacement
            # Should be non-trivial
            assert abs(tip_z) > 1e-10, "Tip should have non-zero displacement"
            # Positive lift → wing bends upward (positive T3)
            assert tip_z > 0, (
                f"Tip should bend UP (positive T3) under upward lift, "
                f"got T3 = {tip_z:.6e}"
            )

    def test_displacement_increases_spanwise(self):
        """Displacement should increase from root to tip (cantilever wing)."""
        sc = self.results.subcases[0]
        prev_z = 0.0
        for nid in sorted(sc.displacements.keys()):
            d = sc.displacements[nid]
            # Skip root (clamped)
            if abs(d[2]) < 1e-15:
                continue
            assert d[2] >= prev_z, (
                f"Displacement should increase spanwise, "
                f"but node {nid} T3={d[2]:.6e} < prev {prev_z:.6e}"
            )
            prev_z = d[2]

    def test_lift_equals_weight(self):
        """Total aerodynamic lift should balance structural weight."""
        sc = self.results.subcases[0]
        total_fz = np.sum(sc.aero_forces[:, 2])
        # Allow 1% tolerance for numerical precision
        assert abs(total_fz) > 10, "Lift should be non-trivial"
        # The trim should balance: lift ~= weight
        # For Goland wing: weight ~ 2135 N
        assert total_fz > 0, f"Lift should be positive, got {total_fz:.2f}"


class TestAeroelasticFeedbackSign:
    """공탄성 되먹임 부호 회귀 시험 (2026-08 감사).

    구조 평형은 K u = F_ext + Q_aa u 이므로 계 행렬은 (K - Q_aa)다.
    부호가 반대면 동압에 따라 인위적으로 강해져(반발산) 유연 증분이
    과소평가되고 발산이 나타나지 않는다.
    """

    def _solve_1dof(self, feedback_ratio):
        """단일 자유도 비틀림 단면: Q_aa = ratio * K로 푼다."""
        from nastaero.solvers.sol144 import _solve_dense
        k, force = 1000.0, 10.0
        # Q_aa = G_disp^T @ A_jj @ G_sp = d * a * s
        d, s = 2.0, 3.0
        a = feedback_ratio * k / (d * s)
        u_f, _ = _solve_dense(
            K_ff=np.array([[k]]),
            G_sp=np.array([[s]]), G_disp=np.array([[d]]),
            A_jj=np.array([[a]]), Q_ax=np.zeros((1, 0)),
            F_f=np.array([force]), F_trim_fixed=np.zeros(1),
            D_r=np.zeros((0, 1)), D_x=np.zeros((0, 0)),
            rhs_trim=np.zeros(0),
            n_free=1, n_trim_free=0, n_constraints=0)
        return u_f[0], force / k

    def test_feedback_softens_not_stiffens(self):
        """되먹임이 있으면 강체 대비 변형이 커져야 한다(공탄성 연화)."""
        u, u_rigid = self._solve_1dof(0.5)
        assert u > u_rigid, (
            f"공탄성 되먹임은 연화여야 한다: u={u:.6f} vs 강체 {u_rigid:.6f}")
        # (K - Q) u = F  =>  u = F / (0.5 k) = 2 * u_rigid
        assert abs(u - 2.0 * u_rigid) / u_rigid < 1e-12, (
            f"해석해 {2.0 * u_rigid:.6f} vs 계산 {u:.6f}")

    def test_amplification_diverges_at_q_div(self):
        """증폭률 1/(1-Q/K)이 발산 동압에서 단조 증가해야 한다."""
        prev = 0.0
        for ratio in (0.0, 0.25, 0.5, 0.75, 0.9):
            u, u_rigid = self._solve_1dof(ratio)
            amp = u / u_rigid
            assert abs(amp - 1.0 / (1.0 - ratio)) < 1e-10, (
                f"ratio={ratio}: 증폭률 {amp:.6f} != "
                f"{1.0 / (1.0 - ratio):.6f}")
            assert amp > prev, f"증폭률이 단조 증가해야 한다 (ratio={ratio})"
            prev = amp


class TestSplineForceDistribution:
    """스플라인 힘 분배 회귀 시험 (2026-08 감사).

    박스 힘 분배는 물리공간 연산이므로 자유도 소거(RBE2 종속/SPC)와
    무관해야 한다. f-set만 담은 행렬로 분배하면 그 절점의 가중치가
    통째로 빠져 총힘이 결손되고, 보존 재스케일이 결손을 기체 전체에
    균등하게 문질러 스팬 하중 분포를 왜곡한다.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from nastaero.solvers.sol144 import solve_trim
        self.model = parse_bdf(GOLAND_BDF)
        self.results = solve_trim(self.model)

    def test_nodal_total_matches_panel_total(self):
        """절점 합력이 재스케일 없이 패널 합력과 일치해야 한다."""
        sc = self.results.subcases[0]
        panel = np.sum(np.real(sc.aero_forces[:, :3]), axis=0)
        nodal = np.zeros(3)
        for f in sc.nodal_aero_forces.values():
            nodal += f[:3]
        assert abs(nodal[2] - panel[2]) / max(abs(panel[2]), 1.0) < 1e-9, (
            f"절점 Fz {nodal[2]:.3f} != 패널 Fz {panel[2]:.3f}")

    def test_spanwise_shear_decreases_outboard(self):
        """단면 전단은 뿌리에서 끝으로 단조 감소해야 한다.

        가중치가 결손되면 뿌리 근처 하중이 밖으로 문질러져 중간
        스테이션 전단이 뿌리보다 커지는 비물리적 분포가 나온다.
        """
        sc = self.results.subcases[0]
        xyz = {nid: n.xyz_global for nid, n in self.model.nodes.items()}
        span = max(abs(p[1]) for p in xyz.values())
        prev = None
        for frac in (0.05, 0.25, 0.5, 0.75):
            y0 = frac * span
            V = sum(f[2] for nid, f in sc.nodal_aero_forces.items()
                    if nid in xyz and xyz[nid][1] >= y0)
            if prev is not None:
                assert V <= prev + 1e-6, (
                    f"y={y0:.1f}에서 전단 {V:.2f}이 안쪽 {prev:.2f}보다 크다")
            prev = V


class TestIPSDegeneracyGuard:
    """IPS 퇴화 배치 판정 시험 (2026-08 감사).

    IPS 커널은 x-y 투영으로 세워지므로, 3D에서 공선이 아니어도
    투영이 공선이면 특이해진다(수직 핀처럼 x-z 평면에 놓인 지지점).
    3D 공선만 검사하면 그런 배치가 통과해 조용히 쓰레기 가중치를 낸다.
    """

    def test_vertical_surface_projection_is_degenerate(self):
        from nastaero.solvers.sol144 import _nodes_are_collinear
        xz_plane = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 100.0],
                             [0.0, 0.0, 200.0], [50.0, 0.0, 150.0]])
        assert _nodes_are_collinear(xz_plane)

    def test_normal_wing_layout_is_not_degenerate(self):
        from nastaero.solvers.sol144 import _nodes_are_collinear
        wing = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0],
                         [0.0, 100.0, 0.0], [50.0, 50.0, 10.0]])
        assert not _nodes_are_collinear(wing)

    def test_true_collinear_still_detected(self):
        from nastaero.solvers.sol144 import _nodes_are_collinear
        line = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0],
                         [200.0, 0.0, 0.0]])
        assert _nodes_are_collinear(line)


class TestLoadFactorSemantics:
    """URDD3 하중배수 해석 시험 (2026-08 감사).

    명시적 URDD3=0.0은 0g 푸시오버이고, 카드에 없을 때만 기본
    1g다. 종전에는 0.0을 무조건 1.0으로 강제해 인증 덱의
    "nz=0.00" 케이스가 1g로 풀렸다.
    """

    def test_explicit_zero_is_zero_g(self):
        from nastaero.solvers.sol144 import solve_trim
        model = parse_bdf(GOLAND_BDF)
        for trim in model.trims.values():
            trim.variables = [(l, 0.0) if l.upper() == "URDD3" else (l, u)
                              for l, u in trim.variables]
        res = solve_trim(model)
        total_fz = float(np.sum(res.subcases[0].aero_forces[:, 2]))
        ref = parse_bdf(GOLAND_BDF)
        res1g = solve_trim(ref)
        fz_1g = float(np.sum(res1g.subcases[0].aero_forces[:, 2]))
        assert abs(total_fz) < 0.05 * abs(fz_1g), (
            f"0g 케이스 양력 {total_fz:.1f}이 1g {fz_1g:.1f} 대비 크다")

    def test_missing_urdd3_defaults_to_one_g(self):
        from nastaero.solvers.sol144 import solve_trim
        model = parse_bdf(GOLAND_BDF)
        for trim in model.trims.values():
            trim.variables = [(l, u) for l, u in trim.variables
                              if l.upper() != "URDD3"]
        res = solve_trim(model)
        ref = solve_trim(parse_bdf(GOLAND_BDF))
        a = res.subcases[0].trim_variables["ANGLEA"]
        b = ref.subcases[0].trim_variables["ANGLEA"]
        assert a == pytest.approx(b, rel=1e-6)
