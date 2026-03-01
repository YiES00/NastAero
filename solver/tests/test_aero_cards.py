"""Tests for aerodynamic BDF card parsers."""
import pytest
import numpy as np
from nastaero.bdf.cards.aero import (
    AERO, AEROS, CAERO1, PAERO1, SPLINE1, SPLINE2,
    AESTAT, AESURF, TRIM, FLFACT, MKAERO1
)
from nastaero.bdf.cards.sets import SET1
from nastaero.bdf.cards.rbe import RBE3


class TestAERO:
    def test_parse(self):
        fields = ["AERO", "0", "50.0", "1.8288", "1.225", "0", "0"]
        a = AERO.from_fields(fields)
        assert a.acsid == 0
        assert a.velocity == pytest.approx(50.0)
        assert a.refc == pytest.approx(1.8288)
        assert a.rhoref == pytest.approx(1.225)

    def test_defaults(self):
        fields = ["AERO", "0", "100.0"]
        a = AERO.from_fields(fields)
        assert a.refc == pytest.approx(1.0)
        assert a.rhoref == pytest.approx(1.225)


class TestAEROS:
    def test_parse(self):
        fields = ["AEROS", "0", "0", "1.8288", "12.192", "22.296"]
        a = AEROS.from_fields(fields)
        assert a.refc == pytest.approx(1.8288)
        assert a.refb == pytest.approx(12.192)
        assert a.refs == pytest.approx(22.296)


class TestCAERO1:
    def test_parse(self):
        fields = ["CAERO1", "1001", "1001", "0", "8", "2", "0", "0", "1",
                  "0.0", "0.0", "0.0", "1.8288",
                  "0.0", "6.096", "0.0", "1.8288"]
        c = CAERO1.from_fields(fields)
        assert c.eid == 1001
        assert c.nspan == 8
        assert c.nchord == 2
        assert c.p1[0] == pytest.approx(0.0)
        assert c.p1[1] == pytest.approx(0.0)
        assert c.chord1 == pytest.approx(1.8288)
        assert c.p4[1] == pytest.approx(6.096)
        assert c.chord4 == pytest.approx(1.8288)

    def test_type(self):
        c = CAERO1()
        assert c.type == "CAERO1"


class TestPAERO1:
    def test_parse(self):
        fields = ["PAERO1", "1001"]
        p = PAERO1.from_fields(fields)
        assert p.pid == 1001
        assert len(p.bodies) == 0


class TestSPLINE1:
    def test_parse(self):
        fields = ["SPLINE1", "100", "1001", "0", "15", "10", "0.0"]
        s = SPLINE1.from_fields(fields)
        assert s.eid == 100
        assert s.caero == 1001
        assert s.box1 == 0
        assert s.box2 == 15
        assert s.setg == 10
        assert s.dz == pytest.approx(0.0)
        assert s.method == "IPS"

    def test_method(self):
        fields = ["SPLINE1", "1", "1001", "0", "15", "10", "0.0", "FPS"]
        s = SPLINE1.from_fields(fields)
        assert s.method == "FPS"


class TestSPLINE2:
    def test_parse(self):
        fields = ["SPLINE2", "200", "1001", "1", "16", "20", "0.0", "1.0", "0"]
        s = SPLINE2.from_fields(fields)
        assert s.eid == 200
        assert s.caero == 1001
        assert s.dtor == pytest.approx(1.0)


class TestAESTAT:
    def test_parse(self):
        fields = ["AESTAT", "501", "ANGLEA"]
        a = AESTAT.from_fields(fields)
        assert a.id == 501
        assert a.label == "ANGLEA"

    def test_upper(self):
        fields = ["AESTAT", "502", "anglea"]
        a = AESTAT.from_fields(fields)
        assert a.label == "ANGLEA"


class TestAESURF:
    def test_parse(self):
        fields = ["AESURF", "601", "AILERON", "0", "101", "0", "102", "0.8"]
        a = AESURF.from_fields(fields)
        assert a.id == 601
        assert a.label == "AILERON"
        assert a.cid1 == 0
        assert a.alid1 == 101
        assert a.eff == pytest.approx(0.8)


class TestTRIM:
    def test_parse(self):
        fields = ["TRIM", "1", "0.3", "1531.25", "URDD3", "0.0"]
        t = TRIM.from_fields(fields)
        assert t.tid == 1
        assert t.mach == pytest.approx(0.3)
        assert t.q == pytest.approx(1531.25)
        assert len(t.variables) == 1
        assert t.variables[0][0] == "URDD3"
        assert t.variables[0][1] == pytest.approx(0.0)

    def test_multiple_vars(self):
        fields = ["TRIM", "2", "0.5", "2000.0",
                  "ANGLEA", "0.05", "ELEV", "-0.01"]
        t = TRIM.from_fields(fields)
        assert len(t.variables) == 2
        assert t.variables[0] == ("ANGLEA", pytest.approx(0.05))
        assert t.variables[1] == ("ELEV", pytest.approx(-0.01))


class TestFLFACT:
    def test_parse(self):
        fields = ["FLFACT", "1", "0.5", "1.0", "1.5", "2.0"]
        f = FLFACT.from_fields(fields)
        assert f.sid == 1
        assert len(f.factors) == 4
        assert f.factors[0] == pytest.approx(0.5)
        assert f.factors[3] == pytest.approx(2.0)


class TestMKAERO1:
    def test_parse(self):
        fields = ["MKAERO1", "0.5", "0.8", "", "", "", "", "", "",
                  "0.01", "0.1", "0.5", "1.0"]
        m = MKAERO1.from_fields(fields)
        assert len(m.machs) == 2
        assert m.machs[0] == pytest.approx(0.5)
        assert len(m.reduced_freqs) == 4
        assert m.reduced_freqs[2] == pytest.approx(0.5)


class TestSET1:
    def test_simple(self):
        fields = ["SET1", "10", "1", "2", "3", "4", "5"]
        s = SET1.from_fields(fields)
        assert s.sid == 10
        assert s.ids == [1, 2, 3, 4, 5]

    def test_thru(self):
        fields = ["SET1", "20", "1", "THRU", "10"]
        s = SET1.from_fields(fields)
        assert s.sid == 20
        assert s.ids == list(range(1, 11))


class TestRBE3:
    def test_parse(self):
        fields = ["RBE3", "100", "", "10", "123", "1.0", "123", "1", "2", "3"]
        r = RBE3.from_fields(fields)
        assert r.eid == 100
        assert r.refgrid == 10
        assert r.refc == "123"
        assert len(r.weight_sets) == 1
        assert r.weight_sets[0][0] == pytest.approx(1.0)
        assert r.weight_sets[0][1] == "123"
        assert r.weight_sets[0][2] == [1, 2, 3]

    def test_type(self):
        r = RBE3()
        assert r.type == "RBE3"

    def test_node_ids(self):
        r = RBE3(refgrid=10, weight_sets=[(1.0, "123", [1, 2, 3])])
        assert r.node_ids == [10, 1, 2, 3]
