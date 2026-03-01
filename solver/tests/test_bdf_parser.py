"""Tests for the BDF parser (parsing complete BDF files)."""
import os
import pytest
from nastaero.bdf.parser import BDFParser

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
