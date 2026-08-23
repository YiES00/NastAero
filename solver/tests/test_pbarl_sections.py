# PBARL/PBEAML 단면 공식의 MSC 규약 적합성 시험 (2026-08 감사)
"""PBARL 단면 치수 해석과 단면 상수 계산 검증.

감사에서 확인된 결함: TUBE의 DIM2를 두께로 읽음(MSC는 내반지름),
I/I1/CHAN1의 치수 순서가 MSCBML0과 불일치, CHAN/CHAN2/HAT/Z 미지원,
NSM 필드를 치수로 삼킴. GACOMP 비교 모델에서 PBARL 30장이 면적
0 이하가 되어 CBAR 316개가 무강성으로 조립되고 있었다.
"""
from __future__ import annotations
import math
import numpy as np
import pytest
from nastaero.bdf.cards.properties import PBARL


def make(type_name, dims):
    fields = ["PBARL", "1", "2", "MSCBML0", type_name, "", "", "", ""]
    fields += [str(d) for d in dims]
    return PBARL.from_fields(fields)


class TestTube:
    def test_dim2_is_inner_radius(self):
        """TUBE의 DIM2는 두께가 아니라 내반지름이다."""
        p = make("TUBE", [10.0, 8.0])
        assert p.A == pytest.approx(math.pi * (100.0 - 64.0))
        assert p.I1 == pytest.approx(math.pi * (10.0**4 - 8.0**4) / 4.0)
        assert p.J == pytest.approx(2.0 * p.I1)

    def test_tube2_is_thickness(self):
        """TUBE2의 DIM2는 벽 두께다 (내반지름 = Ro - t)."""
        p = make("TUBE2", [10.0, 2.0])
        assert p.A == pytest.approx(math.pi * (100.0 - 64.0))


class TestISections:
    def test_i_dim_order(self):
        """I: DIM1=깊이, DIM2=하부폭, DIM3=상부폭, DIM4=웨브두께,
        DIM5=하부두께, DIM6=상부두께."""
        h, wb, wt, tw, tb, tt = 200.0, 120.0, 60.0, 8.0, 12.0, 10.0
        p = make("I", [h, wb, wt, tw, tb, tt])
        hw = h - tb - tt
        assert p.A == pytest.approx(wb * tb + wt * tt + hw * tw)

    def test_i1_derives_flange_thickness(self):
        """I1: 플랜지 두께는 (전체깊이 - 순웨브높이)/2로 유도된다."""
        p = make("I1", [10.2, 1.6, 24.0, 32.0])
        bf, tw, hw, h = 10.2 + 1.6, 1.6, 24.0, 32.0
        tf = (h - hw) / 2.0
        assert p.A == pytest.approx(2 * bf * tf + hw * tw)

    def test_i1_matches_equivalent_i(self):
        """I1은 상하 플랜지가 같은 I와 동일해야 한다."""
        p1 = make("I1", [10.2, 1.6, 24.0, 32.0])
        bf, tw, tf = 11.8, 1.6, 4.0
        p2 = make("I", [32.0, bf, bf, tw, tf, tf])
        assert p1.A == pytest.approx(p2.A)
        assert p1.I1 == pytest.approx(p2.I1)


class TestChannels:
    def test_chan_dims(self):
        """CHAN: DIM1=플랜지폭, DIM2=깊이, DIM3=웨브두께, DIM4=플랜지두께."""
        bf, h, tw, tf = 30.0, 53.0, 2.6208, 2.6208
        p = make("CHAN", [bf, h, tw, tf])
        assert p.A == pytest.approx(2 * bf * tf + (h - 2 * tf) * tw)

    def test_chan1_dims(self):
        """CHAN1: DIM1=웨브 바깥 플랜지폭, DIM2=웨브두께,
        DIM3=순웨브높이, DIM4=전체깊이."""
        p = make("CHAN1", [10.2, 1.6, 18.58, 26.12])
        bf, tw, hw, h = 11.8, 1.6, 18.58, 26.12
        tf = (h - hw) / 2.0
        assert p.A == pytest.approx(2 * bf * tf + (h - 2 * tf) * tw)

    def test_chan2_positive(self):
        p = make("CHAN2", [2.5, 2.0, 40.0, 30.0])
        assert p.A > 0 and p.I1 > 0 and p.I2 > 0


class TestPreviouslyUnsupported:
    @pytest.mark.parametrize("type_name,dims", [
        ("HAT", [42.0, 2.62, 45.0, 33.0]),
        ("Z", [34.5, 2.62, 24.81, 30.05]),
        ("CHAN", [30.0, 53.0, 2.6208, 2.6208]),
        ("CHAN2", [2.62, 2.62, 40.0, 30.0]),
    ])
    def test_yields_positive_section(self, type_name, dims):
        """미지원이던 타입도 양의 단면 상수를 내야 한다 (강성 0 방지)."""
        p = make(type_name, dims)
        assert p.A > 0, f"{type_name} 면적 {p.A}"
        assert p.I1 > 0 and p.I2 > 0 and p.J > 0

    def test_hat_area_matches_thin_wall_sum(self):
        """HAT 면적은 균일 두께 박벽 전개 길이와 일치해야 한다."""
        h, t, wb, wt = 42.0, 2.62, 45.0, 33.0
        p = make("HAT", [h, t, wb, wt])
        assert p.A == pytest.approx(t * (wt + 2 * (h - t) + 2 * wb))


class TestNSM:
    def test_nsm_not_swallowed_into_dims(self):
        """타입별 DIM 개수를 넘는 값은 NSM이지 치수가 아니다."""
        p = make("HAT", [42.0, 1.7472, 45.0, 33.0, 0.5])
        assert len(p.dims) == 4
        assert p.nsm == pytest.approx(0.5)

    def test_box_nsm_does_not_corrupt_section(self):
        """BOX는 DIM 6개 — 7번째 값이 치수로 들어가면 단면이 깨진다."""
        p = make("BOX", [40.0, 60.0, 2.0, 2.0, 2.0, 2.0, 1.25])
        assert len(p.dims) == 6
        assert p.nsm == pytest.approx(1.25)
        assert p.A > 0


class TestUnknownTypeWarns:
    def test_unknown_type_warns(self, caplog):
        """미지원 타입은 조용히 강성 0이 되면 안 된다."""
        import logging
        with caplog.at_level(logging.WARNING):
            p = make("DBOX", [1.0, 2.0, 3.0])
        assert p.A == 0.0
        assert any("DBOX" in r.getMessage() for r in caplog.records)
