"""Tests for BDF field parsing."""
import pytest
from nastaero.bdf.field_parser import (
    nastran_float, nastran_int, nastran_string,
    parse_fixed8, parse_free, detect_format, parse_card_fields,
)


class TestNastranFloat:
    def test_standard_float(self):
        assert nastran_float("1.5") == 1.5

    def test_scientific(self):
        assert nastran_float("1.5e3") == 1500.0

    def test_implicit_exponent_positive(self):
        assert nastran_float("1.5+3") == 1500.0

    def test_implicit_exponent_negative(self):
        assert nastran_float("1.5-3") == pytest.approx(0.0015)

    def test_fortran_d_notation(self):
        assert nastran_float("1.5D3") == 1500.0

    def test_negative_mantissa(self):
        assert nastran_float("-2.5+2") == -250.0

    def test_blank_default(self):
        assert nastran_float("", 0.0) == 0.0
        assert nastran_float("  ", 42.0) == 42.0

    def test_integer_as_float(self):
        assert nastran_float("100") == 100.0

    def test_dot_only(self):
        assert nastran_float("7.") == 7.0

    def test_small_field(self):
        assert nastran_float("8.333-10") == pytest.approx(8.333e-10)


class TestNastranInt:
    def test_basic(self):
        assert nastran_int("42") == 42

    def test_blank(self):
        assert nastran_int("", 0) == 0

    def test_whitespace(self):
        assert nastran_int("  7  ") == 7


class TestFormatDetection:
    def test_free_field(self):
        assert detect_format("GRID,1,,0.0,0.0,0.0") == "free"

    def test_fixed8(self):
        assert detect_format("GRID           1") == "fixed8"

    def test_fixed16(self):
        assert detect_format("GRID*          1") == "fixed16"


class TestParseFixed8:
    def test_basic(self):
        line = "GRID           1       0     0.0     0.0     0.0       0"
        fields = parse_fixed8(line)
        assert fields[0].strip() == "GRID"
        assert fields[1].strip() == "1"


class TestParseFree:
    def test_basic(self):
        fields = parse_free("GRID,1,,0.0,0.0,0.0,,123456")
        assert fields[0] == "GRID"
        assert fields[1] == "1"
        assert fields[7] == "123456"


class TestParseCardFields:
    def test_free_field_card(self):
        lines = ["GRID,10,,1.0,2.0,3.0"]
        fields = parse_card_fields(lines)
        assert fields[0] == "GRID"
        assert fields[1] == "10"
        assert float(fields[3]) == 1.0
