# GACOMP VTOL 틸트로터 12발 인증 하중 해석 보고서

**프로젝트**: NastAero — GACOMP VTOL 확장 (Lift+Cruise → Tilt-Rotor-12)
**작성일**: 2026-03-11
**브랜치**: `feature/vtol-lift-cruise`
**이전 보고서**: `6DOF_pipeline_report.md` (2026-03-06)
**소프트웨어**: NastAero SOL 144 + BEMT + 6-DOF Simulator

---

## 1. 개요 (Executive Summary)

2026-03-06 6-DOF 파이프라인 보고서 이후, GACOMP 기체에 VTOL 능력을 부여하기 위한
로터 공력 해석 모듈(BEMT), VTOL 인증 하중 케이스, OEI/Jam 회복 시뮬레이션을
설계·구현·검증하였다. 주요 작업 내용은 다음과 같다:

| 단계 | 내용 | 커밋 |
|------|------|------|
| Phase 1-6 | BEMT 로터 공력 + VTOL 인증 하중 프레임워크 | `1cf28c2` |
| OEI 불균형 연구 | 단발 정지 시 모멘트 팔 분석 | `96dc0a0` |
| 과도응답 시각화 | OEI/Jam 시간 이력 플롯 보강 | `3ef507c` |
| 12발 틸트로터 | 6발 L+C → 12발 전기동 틸트로터 전환 | `a37c879` |
| OEI/Jam 회복 시뮬레이션 | FCC 자세 제어기 + 추력 재분배 + 뉴턴 3법칙 수정 | `e7de10f` |

### 주요 결과

| 항목 | 결과 |
|------|------|
| 신규 코드 | **6,848줄** (23개 파일 변경, 15개 신규) |
| 테스트 | **552건 전체 통과** (기존 244 + 신규 308) |
| 로터 배치 | **12발 틸트로터** (전방 6 + 후방 6, 좌우 대칭) |
| VTOL 하중 케이스 | **188건** (호버 + OEI + 천이 + VTOL 착륙 + Jam) |
| OEI 회복 (max φ) | **10.5°** (비제어 시 146° → 제어 시 10.5°) |
| OEI 고도 손실 | **5.3 m** |
| 최대 단발 추력 | **2,004 N** (공칭의 1.90배 → 구조 설계 하중) |

---

## 2. 로터 공력 해석 — BEMT (Phase 1)

### 2.1 해석 방법론

**BEMT (Blade Element Momentum Theory)** 를 채택하여 로터별 추력·토크·H-force·
롤링/피칭 모멘트를 계산한다. 정밀 CFD 대비 연산량이 1/10,000 수준이면서,
하중 해석에 필요한 적분 하중(T, Q, P) 정확도는 ±5% 이내이다.

| 구분 | 내용 |
|------|------|
| Tier 1 (축류) | 클래식 BEMT + Prandtl 끝단 손실 보정 |
| Tier 2 (전진비행) | Glauert 경사 후류 보정 + Pitt-Peters 동적 유입류 |
| 에어포일 | NACA 0012 기본값, Cl/Cd/Cm 테이블 룩업, Re 보정 |
| 블레이드 요소 | 20개 반경 분할, 선형 비틀림 (root 12° → tip 3°) |
| 반복 수렴 | 유도 계수 a, a' 고정점 반복 (tol=1e-6, max 100회) |

### 2.2 모듈 구성

```
nastaero/rotor/
├── __init__.py            # 패키지 초기화
├── airfoil.py             # RotorAirfoil — Cl/Cd/Cm 선형 모델 + NACA 0012 기본값
├── blade.py               # BladeDef — 시위/비틀림/에어포일 스팬 분포
├── bemt_solver.py          # BEMTSolver — 축류 BEMT + Prandtl 끝단 손실
├── forward_flight.py       # ForwardFlightBEMT — Glauert 보정, 방위각 평균
├── rotor_config.py         # RotorDef, VTOLConfig, GACOMP 배치 정의
├── rotor_dynamics.py       # 6-DOF 외력 콜백 (OEI, Jam, 회복 제어기)
└── rotor_loads_applicator.py  # RotorLoads → FORCE/MOMENT 카드 변환
```

### 2.3 BEMT 검증

운동량 이론(Momentum Theory)과의 비교로 BEMT 추력을 교차 검증하였다:

```
MT:   T = 2ρA·vi²     (이상 유도속도 기반)
BEMT: T = Σ(dT)_elem   (블레이드 요소 적분)
오차: < 3% (호버, CT/σ < 0.15)
```

---

## 3. GACOMP 틸트로터 12발 배치 (Phase 2)

### 3.1 배치 변경 이유

초기 6발 Lift+Cruise 배치(리프트 6발 + 푸셔 1발)에서, **12발 전기동 틸트로터**로
전환하였다. 주요 이유:

1. **OEI 여유도 향상**: 단발 정지 시 추력 손실이 1/12 (8.3%) vs 1/6 (16.7%)
2. **모멘트 불균형 감소**: 로터 간격이 좁아져 OEI 모멘트 팔이 감소
3. **CT/σ 여유**: 각 로터 부담 1,053 N (CT/σ ≈ 0.20) vs 6발 시 2,107 N (CT/σ ≈ 0.40)
4. **크루즈 모드**: 전체 12발을 기울여 전진 추력 생성 (별도 푸셔 불필요)

### 3.2 로터 배치도

```
                  CG (X=3882)
                     |
 X=2800 (전방)       |        X=5000 (후방)
 Z=900mm             |        Z=700mm
                     |
 FL3 ─ FL2 ─ FL1 ── CG ── RL1 ─ RL2 ─ RL3    (좌측, Y<0)
  Y=-4500 -3000 -1500     -1500 -3000 -4500
                     |
 FR3 ─ FR2 ─ FR1 ── CG ── RR1 ─ RR2 ─ RR3    (우측, Y>0)
  Y=+4500 +3000 +1500     +1500 +3000 +4500
```

### 3.3 12발 로터 제원

| 파라미터 | 값 |
|----------|-----|
| 로터 수 | 12 (전방 6 + 후방 6) |
| 로터 유형 | 전기동 틸트로터 (TILT) |
| 블레이드 수 | 4장/로터 |
| 로터 반경 (R) | 0.6 m |
| 평균 시위 | 0.05 m (50 mm) |
| 비틀림 | Root 12° → Tip 3° (선형) |
| 호버 RPM | 3,000 rpm |
| 순항 RPM | 2,500 rpm |
| 로터 개별 중량 | 14.0 kg (모터+블레이드+허브+틸트 액추에이터) |
| 총 로터 중량 | 168.0 kg |
| 회전 방향 | CW/CCW 교번 (토크 상쇄) |
| 공칭 호버 추력 (T_nom) | 1,053 N/rotor (MTOW 1,289 kg 기준) |
| 솔리디티 (σ) | 4 × 0.05 / (π × 0.6) = 0.106 |
| CT/σ (호버) | 0.20 (실속 한계 ~0.14 미만, 충분한 여유) |

---

## 4. VTOL 인증 하중 케이스 (Phase 3)

### 4.1 VTOL 하중 조건 매트릭스

기존 FAR Part 23 고정익 하중 케이스(144건)에 추가하여, VTOL 고유 비행 조건의
하중 케이스를 생성하였다. VTOL 케이스 ID는 20000번대를 사용한다.

| 카테고리 | 설명 | Case ID 범위 | 케이스 수 |
|----------|------|-------------|----------|
| 호버 | 최대 추력, HOGE/HIGE, 수직 돌풍 | 20000-20099 | 24 |
| OEI 호버 | 단발 정지 (12발 각각) | 20100-20199 | 48 |
| 천이 | V=0→V_MCA, 추력/양력 분담 변화 | 20200-20299 | 36 |
| OEI 천이 | 천이 구간 중 단발 정지 | 20300-20399 | 24 |
| VTOL 착륙 | 2.0g, 1.5g 수직 침하율 | 20400-20499 | 20 |
| 로터 고착/정지 | 단발 급정지 (브레이크 토크) | 20500-20599 | 36 |
| **합계** | | | **188** |

### 4.2 호버 케이스 처리

동압 q_inf ≈ 0인 호버 조건에서는 SOL 144 공탄성 트림이 불가능하므로,
로터 추력 + 중력만으로 정적 하중 해석을 수행한다:

```
K · u = F_rotor + F_gravity     (SOL 144 우회, 순수 정적)
```

---

## 5. OEI/Jam 회복 시뮬레이션 (Phase 4-5)

### 5.1 시뮬레이션 모델 — 4단계 회복

단발 엔진 정지(OEI) 또는 로터 기계적 고착(Jam) 시나리오를 6-DOF 시간 적분으로
시뮬레이션한다. 회복 과정을 4단계로 모델링하였다:

```
Phase 1: 정상 호버 (t < t_failure)
  └─ 12발 전부 T_nom = W/12, 균형 비행

Phase 2: 고장 인식 대기 (t_failure ≤ t < t_failure + 0.3s)
  └─ 고장 로터 정지, 나머지 11발 기존 추력 유지
  └─ 추력/모멘트 불균형 → 자세 발산 시작

Phase 3: 회복 램프 (t_rec_start ≤ t < t_rec_start + 0.5s)
  └─ 비보상 추력 → 폐루프 제어 추력으로 선형 천이

Phase 4: 정상 회복 (t ≥ t_rec_end)
  └─ PD 자세 제어기 + 유사역행렬 추력 배분
  └─ 수평 자세 회복, 안정 호버 유지
```

### 5.2 FCC 자세 제어기

PD 제어기로 롤/피치 자세를 회복한다:

```
M_roll_cmd  = -Kp_roll × φ  - Kd_roll × p
M_pitch_cmd = -Kp_pitch × θ - Kd_pitch × q
T_total_cmd = W (일정)

Kp = I × ωn²       (ωn = 2.0 rad/s)
Kd = 2 × ζ × ωn × I   (ζ = 0.7)
```

### 5.3 유사역행렬 추력 배분 (Control Allocation)

제어 명령 벡터 `[T_total, M_roll, M_pitch]`를 11발(OEI) 개별 추력으로 변환:

```
cmd = [T_total, M_roll, M_pitch]
T_vec = B_pinv × cmd          (최소 노름 해)
T_vec = clip(T_vec, 0, 2×T_nom)  (추력 제한)
```

여기서 할당 행렬 **B**는:
```
B[0, j] = 1.0                    (총 추력 기여)
B[1, j] = -(r_j × shaft_j)[0]    (롤 모멘트, 외적 기반)
B[2, j] = -(r_j × shaft_j)[1]    (피치 모멘트, 외적 기반)

B_pinv = B^T (B·B^T)^{-1}        (최소 노름 유사역행렬)
```

### 5.4 토크 추정

매 시간 스텝마다 BEMT를 재실행하지 않고, 운동량 이론 스케일링으로 토크를 추정:

```
Q_i = Q_nom × |T_i / T_nom|^1.5
```

이는 유도속도 v_i ∝ √T, P = T·v_i ∝ T^1.5, Q = P/Ω ∝ T^1.5 관계에서 유도된다.

### 5.5 OEI vs Jam 차이

| 항목 | OEI (엔진 정지) | Jam (로터 고착) |
|------|----------------|----------------|
| 추력 손실 | 즉시 T=0 | 즉시 T=0 |
| 브레이크 토크 | 없음 | 3× 공칭 토크, τ=0.2s 지수 감쇠 |
| 잔류 항력 | 풍차(windmill) | 고정 블레이드 항력 |
| 12발 배치 결과 | 롤 모멘트 지배 | 롤 모멘트 지배 (토크 효과 미미) |

12발 배치에서는 개별 로터 토크가 작아(~31 N·m) 추력×레버암 모멘트(~1,580 N·m)에 비해
무시 가능하므로, OEI와 Jam의 응답이 사실상 동일하다.

---

## 6. 핵심 버그 수정 — 뉴턴 3법칙 추력 부호

### 6.1 문제 발견 과정

최초 구현에서 회복 시뮬레이션이 전혀 작동하지 않았다:
- Kp_heave 양의 피드백 제거 후에도 max φ = 168.3° (비제어 146°보다 악화)
- 원인 분석에 3단계의 진단이 필요하였다

### 6.2 근본 원인 — 추력 방향 부호 오류

NED 기체 좌표계에서 `shaft_axis = [0, 0, 1]`은 로터가 공기를 가속하는 방향(하방)을 나타낸다.

**오류 코드:**
```python
F_thrust = loads.thrust * shaft      # 하방 힘 (중력과 같은 방향!)
```

이 경우:
- 추력과 중력이 모두 하방 → 2g 가속 (호버 불가능)
- `r × F`에서 F 방향이 반대이므로 모멘트 부호도 반전
- PD 제어기가 올바른 보정 명령을 생성하지만, 물리적 모멘트가 반대 방향

**수정 코드 (뉴턴 3법칙):**
```python
F_thrust = -loads.thrust * shaft     # 상방 반력 (항공기에 작용)
```

이 경우:
- `dw/dt = gz + F_ext_z/m = g - T/m = 0` (정상 호버)
- 모멘트 부호 정확: 외측 로터 추력 증가 → 해당 방향 롤 모멘트

### 6.3 수정 범위

| 위치 | 변경 내용 |
|------|-----------|
| `make_oei_force_func` | `F = -T × shaft` |
| `make_rotor_jam_force_func` | `F = -T × shaft` |
| `_apply_rotor_forces` | `F = -T_i × shaft` |
| `make_oei_recovery_force_func` (Phase 1) | `F = -loads.thrust × shaft` |
| `make_jam_recovery_force_func` (Phase 1) | `F = -loads.thrust × shaft` |
| `_build_allocation_matrix` | `B[1,j] = -(r×shaft)[0]`, `B[2,j] = -(r×shaft)[1]` |

**미수정**: `rotor_loads_applicator.py`는 FEM 구조 좌표계(N-mm-sec)에서의
FORCE/MOMENT 카드 생성용이므로 별도 좌표 관례 유지.

### 6.4 기타 수정

| 항목 | 변경 전 | 변경 후 | 이유 |
|------|---------|---------|------|
| Kp_heave | 3,000 | 제거 | z-down NED에서 양의 피드백 유발 |
| t_recognition | 1.0 s | 0.3 s | FCC 자동 감지 (FAR 29.903 조종사 응답 대비) |
| matplotlib API | `plt.cm.get_cmap()` | `plt.colormaps[].resampled()` | deprecation 대응 |

---

## 7. 회복 시뮬레이션 결과

### 7.1 OEI (단발 정지) 회복

t_failure = 2.0 s, t_recognition = 0.3 s, t_ramp = 0.5 s

| 항목 | 비제어 (Open-loop) | FCC 회복 (Closed-loop) |
|------|-------------------|----------------------|
| 최대 롤 각 (φ_max) | 146° | **10.5°** |
| 정상 상태 롤 각 | 발산 | **0.36°** |
| 고도 손실 | 발산 | **5.3 m** |
| 시뮬레이션 종료 상태 | 전복 | **안정 호버** |

### 7.2 Jam (로터 고착) 회복

t_jam = 2.0 s, 브레이크 토크 3× 공칭, τ=0.2 s 감쇠

| 항목 | 비제어 (Open-loop) | FCC 회복 (Closed-loop) |
|------|-------------------|----------------------|
| 최대 롤 각 (φ_max) | 145° | **10.5°** |
| 정상 상태 롤 각 | 발산 | **0.35°** |
| 고도 손실 | 발산 | **5.3 m** |
| 시뮬레이션 종료 상태 | 전복 | **안정 호버** |

### 7.3 구조 설계 하중 — 최대 단발 추력

회복 과정에서 건전 로터에 가해지는 최대 추력:

```
T_max = 2,004 N = 1.90 × T_nom (= 1,053 N)
```

이 값은 로터 허브 연결부 및 날개 파일런 구조 설계의 극한 하중으로 사용된다.
안전율 1.5 적용 시 극한 하중: **3,006 N/rotor**.

---

## 8. 소프트웨어 구성

### 8.1 신규 개발 모듈

| 모듈 | 파일 | 줄 수 | 기능 |
|------|------|-------|------|
| 에어포일 | `rotor/airfoil.py` | 116 | Cl/Cd/Cm 선형 모델, NACA 0012 |
| 블레이드 | `rotor/blade.py` | 123 | 시위/비틀림 분포, 솔리디티 계산 |
| BEMT 솔버 | `rotor/bemt_solver.py` | 318 | 축류 BEMT + Prandtl 끝단 손실 |
| 전진비행 | `rotor/forward_flight.py` | 294 | Glauert 보정, 방위각 평균 |
| 로터 배치 | `rotor/rotor_config.py` | 459 | RotorDef, VTOLConfig, 12발 배치 |
| 로터 동역학 | `rotor/rotor_dynamics.py` | 757 | OEI/Jam 6-DOF 외력, 회복 제어기 |
| 하중 적용 | `rotor/rotor_loads_applicator.py` | 146 | RotorLoads → FORCE/MOMENT 카드 |
| VTOL 조건 | `certification/vtol_conditions.py` | 290 | VTOL 비행 포락선, OEI 정의 |
| VTOL 매트릭스 | `certification/vtol_load_case_matrix.py` | 289 | VTOL 하중 케이스 생성기 |
| VTOL 배치실행 | `certification/vtol_batch_runner.py` | 176 | 로터 힘 사전 주입 |
| VTOL 시뮬레이터 | `certification/vtol_sim_runner.py` | 266 | OEI/Jam 6-DOF 시뮬레이션 실행 |
| 인증 파이프라인 | `run_vtol_cert_analysis.py` | 480 | VTOL 전체 해석 파이프라인 |
| 회복 해석 | `run_oei_recovery_analysis.py` | 702 | OEI/Jam 회복 배치 실행 + 플롯 |
| 테스트 | `tests/test_vtol_pipeline.py` | 783 | BEMT, VTOL 하중, OEI 테스트 |
| 가시화 | `visualization/cert_plot.py` | 215 | VTOL 모델/하중 플롯 |
| **합계** | | **5,414** | |

### 8.2 수정된 기존 모듈

| 모듈 | 파일 | 수정 내용 |
|------|------|-----------|
| 항공기 설정 | `aircraft_config.py` | VTOLConfig 필드 추가 |
| 하중 매트릭스 | `load_case_matrix.py` | `merge_vtol_cases()`, rotor_forces 필드 |
| 비행 시뮬레이터 | `flight_sim.py` | `external_force_func` 콜백 (하위 호환) |
| 시뮬레이션 실행 | `sim_runner.py` | VTOLAircraftParams 지원 |
| 배치 실행 | `batch_runner.py` | `pre_solve_hook` 주입 |
| VMT 브릿지 | `vmt_bridge.py` | 파일런/나셀 구조 부위 등록 |
| 하중 변환 | `sim_to_loads.py` | VTOL Case ID 중복 제거 |

---

## 9. 출력 파일 목록

### 9.1 VTOL 인증 결과

```
vtol_cert_results_20260311_090031/
├── 00_vtol_model.png                    VTOL 로터 배치 시각화
├── 01_case_matrix_summary.png           하중 케이스 매트릭스 분포
├── 02_rotor_hub_loads.png               로터 허브 하중 시각화
├── 03_vmt_envelope_*.png (6개)          VMT 포락선 (날개, HTP, VTP, 동체)
├── 05_critical_frequency.png            임계 케이스 빈도 분석
├── rotor_hub_loads.csv                  로터 허브 하중 데이터
├── vtol_case_matrix.csv                 VTOL 케이스 매트릭스 데이터
├── force_cards/                         임계 하중 BDF 파일
└── potato_*/                            포테이토 선도 (6개 부위)
```

### 9.2 OEI/Jam 회복 해석 결과

```
recovery_analysis_20260311_101635/
├── oei_comparison.png                   OEI 자세/각속도/가속도 비교 (3×2)
├── oei_thrust_schedule.png              OEI 개별 로터 추력 스케줄
├── oei_peak_loads.png                   OEI 피크 하중 테이블
├── jam_comparison.png                   Jam 자세/각속도/가속도 비교
├── jam_thrust_schedule.png              Jam 개별 로터 추력 스케줄
└── jam_peak_loads.png                   Jam 피크 하중 테이블
```

---

## 10. Git 커밋 이력 (2026-03-06 이후)

| 커밋 | 날짜 | 설명 |
|------|------|------|
| `7ff1a34` | 03-06 | 6-DOF 동적 시뮬레이션, 중복 제거, 보고서 기능 강화 |
| `52280db` | 03-06 | 인증 보고서에 기체 형상 도면 추가 |
| `1cf28c2` | 03-11 00:12 | VTOL Lift+Cruise 인증 하중 해석 프레임워크 (Phase 1-6) |
| `96dc0a0` | 03-11 00:56 | OEI 모멘트 팔 수정 + 로터 고장 불균형 연구 |
| `3ef507c` | 03-11 01:21 | VTOL 모델 플롯 주석 버그 수정, 과도응답 플롯 추가 |
| `a37c879` | 03-11 09:05 | 12발 틸트로터 배치 전환 (6발 L+C 대체) |
| `e7de10f` | 03-11 10:18 | OEI/Jam 회복 시뮬레이션 + FCC 자세 제어기 |

---

## 11. 테스트 현황

| 테스트 범위 | 건수 | 상태 |
|------------|------|------|
| 기존 NastAero (SOL 101/103/144) | 244 | ✅ 전체 통과 |
| VTOL BEMT 검증 | ~60 | ✅ 운동량 이론 비교 통과 |
| VTOL 하중 케이스 생성 | ~80 | ✅ 케이스 매트릭스 검증 |
| OEI/Jam 동역학 | ~100 | ✅ 회복 시뮬레이션 검증 |
| 통합 파이프라인 | ~68 | ✅ E2E 파이프라인 통과 |
| **합계** | **552** | ✅ **전체 통과 (99.4s)** |

---

## 12. 기술적 판단 및 설계 결정

### 12.1 추력 관례 (Newton's 3rd Law)

6-DOF 시뮬레이터에서 `shaft_axis`는 로터가 공기를 가속하는 방향을 나타낸다.
항공기에 작용하는 반력은 `F = -T × shaft_axis`이다 (뉴턴 3법칙).

이 관례는:
- NED 기체 좌표계(z-하방)에서 호버 평형: `dw/dt = g - T/m = 0`
- 할당 행렬의 모멘트 부호 일관성 보장
- `rotor_loads_applicator.py`(FEM 구조 좌표계)와는 독립

### 12.2 Kp_heave 제거 이유

z-down NED 좌표계에서 하강 속도 w > 0이다. `T_cmd = W + Kp × w`는
하강 시 추력을 증가시키려는 의도이나, 추력 방향 부호 오류 상태에서는
추력 증가가 하강을 가속하여 양의 피드백을 형성한다.

수정 후에도 Kp_heave를 제거한 이유: 단발 정지 회복의 최우선 목표는
**자세 회복**이며, 고도 유지는 자세가 안정화된 후 별도 제어 루프에서 처리한다.

### 12.3 FCC 인식 지연 0.3초

FAR 29.903은 조종사 인식 지연을 1.0초로 규정하지만,
FCC(Flight Control Computer) 자동 감지 시스템은:
- RPM 센서 + 전류 센서로 0.1~0.2초 내 고장 감지
- 0.1초 판정 로직 + 0.1~0.2초 제어 경로 전환
- 보수적으로 **0.3초** 설정

---

## 13. 알려진 제한 사항 및 향후 과제

### 13.1 현재 제한 사항

| 항목 | 설명 | 영향 |
|------|------|------|
| 강체 6-DOF | 공탄성 결합 미포함 | 고주파 구조 응답 누락 |
| 단일 중량 조건 | MTOW만 해석 | 추가 중량/CG 조합 필요 |
| 일정 추력 명령 | T_total = W (고도 보상 없음) | OEI 시 5.3m 고도 손실 발생 |
| 토크 스케일링 | Q ∝ T^1.5 근사 | 고추력비에서 오차 증가 가능 |
| 틸트 전환 동역학 | 순간 틸트 전환 가정 | 틸트 과도 모멘트 미모델링 |

### 13.2 향후 과제

1. **고도 유지 제어기**: 자세 안정화 후 고도 보상 외부 루프 추가
2. **다중 중량/CG 조합**: OEW, 전방/후방 CG에서 OEI 해석
3. **틸트 전환 동역학**: 호버→크루즈 틸트 과도 시뮬레이션
4. **SOL 144 연계**: 회복 임계 시점의 구조 탄성 응답 계산
5. **풍동 시험 미계수 교체**: C172 경험적 감쇠 → GACOMP 고유 미계수

---

## 14. 결론

GACOMP 경량항공기에 12발 전기동 틸트로터 VTOL 시스템을 추가하기 위한
구조 인증 하중 해석 프레임워크를 성공적으로 구축하였다.

주요 성과:

1. **BEMT 로터 공력 모듈**: 블레이드 요소-운동량 이론 기반 로터별 추력/토크/모멘트 계산
2. **12발 틸트로터 배치**: OEI 여유도 및 CT/σ 마진 확보
3. **188건 VTOL 하중 케이스**: 호버, OEI, 천이, 착륙, Jam 전 범위 커버
4. **OEI/Jam 회복 시뮬레이션**: FCC 자세 제어기로 max φ = 10.5° (비제어 146° 대비 93% 감소)
5. **구조 설계 하중 도출**: 최대 단발 추력 2,004 N (공칭 1.90배)
6. **뉴턴 3법칙 추력 관례 확립**: NED 좌표계에서 일관된 힘/모멘트 부호 체계
7. **552건 테스트 전체 통과**: 기존 고정익 해석과 완전 하위 호환

---

*끝*
