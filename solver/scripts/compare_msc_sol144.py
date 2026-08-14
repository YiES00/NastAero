# MSC Nastran SOL 144 F06와 NastAero 트림 결과를 자동 대조하는 비교 스크립트
"""ILC-8 MSC 비교 덱의 트림 변수를 두 솔버 간 자동 대조한다.

사용법:
    python scripts/compare_msc_sol144.py <msc.f06> [--naero <result.naero>]

.naero 결과가 없으면 tests/validation/ILC8/ilc8_msc_sol144.bdf를 직접
풀어서 비교한다.

규약 매핑 (물리 조종면 값으로 비교):
- AELINK: MSC는 u_D = -sum(C_i * u_i), NastAero는 u_D = +sum(C_i * u_i).
  마스터 변수(ELEV/RUD)의 부호가 반대로 인쇄되므로, 링크된 물리 조종면
  (ELEVR/ELEVL)의 값을 비교 기준으로 삼는다.
  NastAero: ELEVR = ELEV + RUD, ELEVL = ELEV - RUD.
  MSC: F06의 ELEVR/ELEVL LINKED 행을 그대로 읽는다.
- SC7 사이드슬립: 두 코드의 SIDES 워시 부호 규약이 달라 비대칭(러더)
  성분의 부호가 반대로 나온다. 크기 |a| = |ELEVR-ELEVL|/2 로 비교한다.
- MSC 자유 URDD2/4/6 인쇄값은 측방 강체질량 결합이 미소해 수치적으로
  불안정한 출력 특이점이다(기체 전체 계수표의 실제 관성 CY는 ~0).
  비교 대상에서 제외한다.
"""
from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DECK = REPO / "tests" / "validation" / "ILC8" / "ilc8_msc_sol144.bdf"


def parse_msc_f06(path: Path) -> dict:
    """F06에서 서브케이스별 AEROELASTIC TRIM VARIABLES 블록을 파싱한다."""
    subcases: dict = {}
    current_sc = None
    in_block = False
    row_re = re.compile(
        r"^\s*(\d+)?\s+([A-Z][A-Z0-9]*)\s+"
        r"(RIGID BODY|CONTROL SURFACE|GENERAL CONTROL)\s+"
        r"(FIXED|FREE|LINKED)\s+([-+]?\d\.\d+E[-+]\d+)")
    sc_re = re.compile(r"SUBCASE\s+(\d+)\s*$")

    for line in path.read_text(errors="replace").splitlines():
        m = sc_re.search(line)
        if m:
            current_sc = int(m.group(1))
        if "AEROELASTIC TRIM VARIABLES" in line:
            in_block = True
            continue
        if in_block:
            m = row_re.match(line)
            if m:
                label, status, value = m.group(2), m.group(4), float(m.group(5))
                sc = subcases.setdefault(current_sc, {})
                sc[label] = (value, status)
            elif "CONTROL SURFACE POSITION" in line:
                in_block = False
    return subcases


def solve_nastaero(naero_path: Path | None) -> dict:
    """NastAero 결과(.naero)를 로드하거나 덱을 직접 풀어 트림 변수를 얻는다."""
    if naero_path is None:
        tmp = Path(tempfile.mkdtemp(prefix="msc_cmp_"))
        out = tmp / "ilc8_msc_sol144.naero"
        subprocess.run(
            [sys.executable, "-m", "nastaero", str(DECK),
             "--save-results", str(out)],
            cwd=REPO, check=True, capture_output=True)
        naero_path = out
    from nastaero.output.result_io import load_results
    results, _ = load_results(str(naero_path))
    out_map = {}
    for sc in results.subcases:
        out_map[sc.subcase_id] = dict(sc.trim_variables or {})
    return out_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("f06", type=Path)
    ap.add_argument("--naero", type=Path, default=None)
    args = ap.parse_args()

    msc = parse_msc_f06(args.f06)
    nast = solve_nastaero(args.naero)

    r2d = math.degrees(1.0)
    print(f"{'SC':>2} {'quantity':<14} {'MSC':>12} {'NastAero':>12} "
          f"{'diff%':>8}")
    print("-" * 55)
    for sc_id in sorted(msc):
        m, n = msc[sc_id], nast.get(sc_id, {})
        elev_n = n.get("ELEV", 0.0)
        rud_n = n.get("RUD", 0.0)
        rows = [
            ("alpha [deg]", m.get("ANGLEA", (0, ""))[0] * r2d,
             n.get("ANGLEA", 0.0) * r2d),
            ("ELEVR [rad]", m.get("ELEVR", (0, ""))[0], elev_n + rud_n),
            ("ELEVL [rad]", m.get("ELEVL", (0, ""))[0], elev_n - rud_n),
        ]
        if abs(rud_n) > 1e-9 or abs(m.get("RUD", (0, ""))[0]) > 1e-9:
            a_m = 0.5 * abs(m.get("ELEVR", (0, ""))[0]
                            - m.get("ELEVL", (0, ""))[0])
            rows.append(("|rudder| [rad]", a_m, abs(rud_n)))
        for name, vm, vn in rows:
            ref = max(abs(vm), abs(vn), 1e-12)
            print(f"{sc_id:>2} {name:<14} {vm:>12.6f} {vn:>12.6f} "
                  f"{100.0 * abs(vm - vn) / ref:>7.1f}%")
        print()


if __name__ == "__main__":
    main()
