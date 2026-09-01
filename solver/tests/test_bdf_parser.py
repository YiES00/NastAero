"""Tests for the BDF parser (parsing complete BDF files)."""
import os
import pytest
from ascent_load.bdf.parser import BDFParser

VALIDATION_DIR = os.path.join(os.path.dirname(__file__), "validation")
CANTILEVER_BDF = os.path.join(VALIDATION_DIR, "cantilever_beam", "cantilever.bdf")
PLATE_MODES_BDF = os.path.join(VALIDATION_DIR, "plate_modes", "plate_modes.bdf")
BEAM_MODES_BDF = os.path.join(VALIDATION_DIR, "cantilever_beam", "beam_modes.bdf")


def parse_bdf(filepath):
    """Helper to call the parser."""
    parser = BDFParser()
    return parser.parse(filepath)


class TestCantileverParsing:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = parse_bdf(CANTILEVER_BDF)

    def test_sol(self):
        assert self.model.sol == 101

    def test_nodes(self):
        assert len(self.model.nodes) == 11

    def test_node_coords(self):
        assert self.model.nodes[1].xyz[0] == pytest.approx(0.0)
        assert self.model.nodes[11].xyz[0] == pytest.approx(1.0)

    def test_elements(self):
        assert len(self.model.elements) == 10

    def test_element_type(self):
        assert self.model.elements[1].type == "CBAR"

    def test_property(self):
        assert 1 in self.model.properties
        assert self.model.properties[1].A == pytest.approx(1.0e-4)

    def test_material(self):
        assert 1 in self.model.materials
        assert self.model.materials[1].E == pytest.approx(7.0e10)

    def test_spc(self):
        assert 1 in self.model.spcs
        assert len(self.model.spcs[1]) >= 1

    def test_force(self):
        assert 1 in self.model.loads
        force = self.model.loads[1][0]
        assert force.type == "FORCE"
        assert force.mag == pytest.approx(100.0)

    def test_cross_reference(self):
        elem = self.model.elements[1]
        assert elem.property_ref is not None
        assert elem.property_ref.material_ref is not None


class TestPlateModelParsing:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = parse_bdf(PLATE_MODES_BDF)

    def test_sol(self):
        assert self.model.sol == 103

    def test_nodes(self):
        assert len(self.model.nodes) == 25

    def test_elements(self):
        assert len(self.model.elements) == 16

    def test_element_type(self):
        assert self.model.elements[1].type == "CQUAD4"

    def test_eigrl(self):
        assert 10 in self.model.eigrls
        assert self.model.eigrls[10].nd == 10

    def test_pshell(self):
        assert self.model.properties[1].t == pytest.approx(0.01)


class TestBeamModesParsing:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = parse_bdf(BEAM_MODES_BDF)

    def test_sol(self):
        assert self.model.sol == 103

    def test_eigrl(self):
        assert 1 in self.model.eigrls
        assert self.model.eigrls[1].nd == 6


class TestDMIParsing:
    """DMI 직접 행렬 입력 — W2GJ 캠버/비틀림 다운워시 조립 검증."""

    def _parse(self, tmp_path, text):
        p = tmp_path / "dmi.bdf"
        p.write_text("SOL 144\nCEND\nBEGIN BULK\n" + text + "ENDDATA\n")
        return parse_bdf(str(p))

    def test_w2gj_small_field(self, tmp_path):
        deck = (
            "DMI     W2GJ           0       2       1       0"
            "               6       1\n"
            "DMI     W2GJ           1       1 -0.0576  0.0268   0.036"
            "  0.0156\n"
            "          0.0104  0.0275\n"
        )
        m = self._parse(tmp_path, deck)
        d = m.dmis["W2GJ"]
        assert d.matrix.shape == (6, 1)
        import numpy as np
        assert np.allclose(
            d.matrix[:, 0],
            [-0.0576, 0.0268, 0.036, 0.0156, 0.0104, 0.0275])

    def test_sparse_row_restart(self, tmp_path):
        # 컬럼 카드 중간의 정수 필드는 새 행 인덱스로 해석돼야 한다
        deck = (
            "DMI     A              0       2       1       0"
            "               5       2\n"
            "DMI     A              1       1     1.5       4     4.5\n"
            "DMI     A              2       3     9.9\n"
        )
        m = self._parse(tmp_path, deck)
        a = m.dmis["A"].matrix
        assert a[0, 0] == 1.5      # (1,1)
        assert a[3, 0] == 4.5      # 정수 4 → 행 재시작
        assert a[1, 0] == 0.0
        assert a[2, 1] == 9.9      # 2번째 컬럼
    def test_column_before_header_ignored(self, tmp_path):
        deck = "DMI     B              1       1     1.0\n"
        m = self._parse(tmp_path, deck)
        assert "B" not in m.dmis
