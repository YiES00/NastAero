# NastAero Solver - Implementation Plan

## Context

NastAero 리포지토리(D:\git\NastAero) 내에 `solver/` 서브디렉토리로 구조해석 FEA 솔버를 새로 개발합니다. MSC Nastran BDF 호환 I/O를 갖추고, 정적해석(SOL 101), 모달해석(SOL 103), 에어로일래스티시티(SOL 144/145/146)를 구현합니다. 향후 하중해석 프로그램으로 확장 가능한 아키텍처를 설계합니다.

## Tech Stack
- **Python**: I/O 파싱, 전처리, 후처리, 오케스트레이션
- **C++ (pybind11)**: 요소 강성행렬, 조립, 선형대수, DLM 커널
- **Eigen**: 희소행렬, 고유치 솔버 (C++ header-only)
- **Spectra**: 대규모 고유치 문제 (Lanczos, shift-invert)
- **SciPy**: Python fallback 솔버
- **scikit-build-core + CMake**: 빌드 시스템

## Project Structure

```
D:\git\NastAero\solver\
├── CMakeLists.txt
├── pyproject.toml
├── nastaero/                    # Python 패키지
│   ├── __init__.py
│   ├── __main__.py              # CLI: python -m nastaero input.bdf
│   ├── config.py
│   ├── bdf/                     # BDF 파서
│   │   ├── parser.py            # 메인 파서 (INCLUDE, continuation 처리)
│   │   ├── cards.py             # 필드 파싱 (fixed-8, fixed-16, free-field)
│   │   ├── bulk_data.py         # 카드 디스패치
│   │   ├── executive_control.py # SOL, CEND 파싱
│   │   ├── case_control.py      # SUBCASE, LOAD, SPC, METHOD 등
│   │   ├── model.py             # BDFModel 컨테이너
│   │   └── cards/               # 개별 카드 파서
│   │       ├── grid.py, coord.py, elements.py, properties.py
│   │       ├── materials.py, loads.py, constraints.py
│   │       ├── aero.py, aero_dynamic.py, eigrl.py
│   │       ├── tables.py, rbe.py, mass.py, param.py
│   ├── fem/                     # 유한요소 코어
│   │   ├── model.py             # FEModel: DOF 매핑, 조립 시스템
│   │   ├── dof_manager.py       # 절점 → 전역 DOF 인덱스
│   │   ├── assembly.py          # 조립 오케스트레이터
│   │   ├── boundary.py          # SPC/MPC 적용
│   │   ├── load_vector.py       # 하중 벡터 조립
│   │   ├── coordinate_systems.py
│   │   └── mass_matrix.py
│   ├── elements/                # 요소 정식화
│   │   ├── base.py, rod.py, bar.py, beam.py
│   │   ├── tria3.py, quad4.py, tria6.py, quad8.py
│   │   ├── tetra4.py, hexa8.py, tetra10.py, hexa20.py
│   │   ├── rbe2.py, rbe3.py, conm2.py
│   │   ├── gauss.py, shape_functions.py
│   ├── solvers/                 # 해석 시퀀스
│   │   ├── base.py, sol101.py, sol103.py
│   │   ├── sol144.py, sol145.py, sol146.py
│   │   ├── linear_solver.py, eigen_solver.py
│   ├── aero/                    # 공력탄성 모듈
│   │   ├── dlm.py               # Doublet-Lattice Method
│   │   ├── aic.py               # AIC 행렬 계산
│   │   ├── spline.py            # 구조-공력 메시 보간
│   │   ├── flutter_solver.py    # k-method, p-k method
│   │   ├── rfa.py               # Roger's Rational Function Approximation
│   │   ├── gust.py, trim.py
│   ├── output/                  # 출력
│   │   ├── f06_writer.py, op2_writer.py, punch_writer.py
│   │   ├── vtk_writer.py, csv_writer.py, result_data.py
│   └── loads_analysis/          # 향후 하중해석 확장
│       ├── monitor_points.py, load_envelopes.py, dyn_loads.py
├── csrc/                        # C++ 소스
│   ├── CMakeLists.txt
│   ├── include/nastaero/        # 헤더
│   ├── src/
│   │   ├── assembly.cpp
│   │   ├── elements/            # bar, beam, tria3, quad4, tetra4, hexa8
│   │   ├── solvers/             # direct_solver, eigen_solver
│   │   └── aero/                # dlm_kernel, aic_compute
│   └── bindings/                # pybind11 바인딩
├── extern/                      # 서드파티 (git submodule)
│   ├── eigen/, pybind11/, spectra/
├── tests/                       # 테스트 스위트
│   ├── test_bdf_parser/, test_elements/, test_solvers/, test_aero/
│   └── validation/              # Goland wing, AGARD 445.6 등
├── docs/ 및 examples/
```

## Implementation Phases (Phase 1부터 시작)

### Phase 1: Foundation - BDF 파서 + SOL 101 (현재)

1. 프로젝트 초기 구조 생성 (디렉토리, pyproject.toml, CMakeLists.txt)
2. BDF 파서 코어: fixed-8/free-field 파싱, continuation, INCLUDE
3. Executive/Case control 파싱 (SOL, SUBCASE, SPC, LOAD, DISPLACEMENT)
4. 기본 카드: GRID, CORD2R, CBAR/PBAR, MAT1, SPC1, FORCE, MOMENT, GRAV, LOAD
5. DOF 매니저 및 좌표계 변환
6. CBAR 요소 강성행렬 (Python, 12x12 Euler-Bernoulli beam)
7. 전역 강성행렬 조립 (SciPy sparse COO→CSC)
8. SPC 적용 (파티션 방식)
9. SOL 101 직접 풀이 (scipy.sparse.linalg.spsolve)
10. F06 출력 (변위, SPC 반력)
11. 검증: 캔틸레버 빔 해석

### Phase 2: Shell 요소 + SOL 103
- CQUAD4/CTRIA3 (MITC4/MITC3), 질량행렬, CONM2, RBE2
- EIGRL 파싱, SciPy eigsh 기반 고유치 풀이
- 검증: 평판 고유진동수

### Phase 3: C++ 코어 포팅
- pybind11 빌드 시스템, 요소 강성/조립/솔버 C++ 구현
- CTETRA, CHEXA 솔리드 요소, RBE3
- Eigen SparseLDLT, Spectra Lanczos

### Phase 4: DLM + SOL 144
- CAERO1 패널 생성, DLM 커널, AIC 행렬
- SPLINE1/2 보간, 구조-공력 변환행렬
- 정적 공탄성 트림

### Phase 5: SOL 145 (Flutter)
- FLUTTER/FLFACT/MKAERO 파싱
- 일반화 AIC, k-method/p-k method
- V-g/V-f 플롯, Goland wing 검증

### Phase 6: SOL 146 (Dynamic Aeroelastic Response)
- Roger's RFA, 상태공간 정식화, 시간적분
- 돌풍 모델 (1-cosine, von Karman PSD)

### Phase 7: 생산 안정화
- OP2 출력, VTK 시각화, 복합재 (PCOMP), 에러 처리, CI/CD

## Key Algorithms

- **CBAR**: 12x12 Euler-Bernoulli beam (EA/L, 12EIz/L³, GJ/L)
- **CQUAD4**: MITC4 formulation (shear locking 방지), 2x2 Gauss 적분
- **CTRIA3**: CST membrane + DKT bending + drilling DOF stabilization
- **DLM**: Landahl/Rodden-Taylor-McIntosh 커널, Desmarais 근사
- **Eigenvalue**: Shift-invert Lanczos (Spectra), (K-σM)⁻¹M 변환
- **Flutter p-k**: 반복수렴 (k→Q_hh→eigenvalue→ω→k_new)
- **RFA**: Roger's 형태, 최소자승 피팅, 상태공간 변환

## Verification

1. **Unit tests**: 각 요소 해석해 비교 (cantilever δ=PL³/3EI, plate Navier 급수)
2. **Integration tests**: 전체 BDF→결과 파이프라인, pyNastran으로 결과 비교
3. **Benchmark**: Goland wing flutter (V_f≈137.2 m/s), AGARD 445.6 flutter boundary
4. **명령어**: `pytest tests/ -v` (전체), `python -m nastaero tests/validation/cantilever_beam/cantilever.bdf`

## 완료된 Phase

### Phase 1+2 완료 (63 tests passing)
- BDF 파서 (fixed-8/16/free-field, 17 card types)
- CBAR/CQUAD4/CTRIA3 요소, DOF 매니저, 전역 조립
- SOL 101 (정적), SOL 103 (모달), F06 출력
- 검증: 캔틸레버 빔 (<0.2%), 빔 모드 (<5%), 평판 모드

---

## 지금 구현할 것: Phase 3a + Phase 4 (DLM + SOL 144)

### Context
Phase 1+2가 완료되어 구조해석 기반이 갖춰졌으므로, C++ 포팅(기존 Phase 3)은 건너뛰고 에어로탄성 핵심 기능인 **DLM (Doublet-Lattice Method) + SOL 144 (정적 공탄성 트림)**을 Python으로 구현합니다. 향후 하중해석 프로그램 확장의 핵심 기능입니다.

### Step 1: 추가 구조 카드 (Phase 3a)
- `nastaero/bdf/cards/sets.py` → SET1 카드 (THRU 지원)
- `nastaero/bdf/cards/rbe.py` 확장 → RBE3 (가중 강체 요소, 공력하중 분배에 필수)
- `nastaero/bdf/bulk_data.py` 업데이트

### Step 2: 에어로 BDF 카드 파서
- `nastaero/bdf/cards/aero.py` (신규 파일, ~300줄)
  - AERO: 기준 코드/속도/밀도/대칭
  - AEROS: 정적 공탄성 기준 (코드/스팬/면적)
  - CAERO1: 공력 패널 정의 (LE 좌표, 코드, spanwise/chordwise 분할)
  - PAERO1: 공력 프로퍼티
  - SPLINE1: Surface spline (무한 평판 스플라인)
  - SPLINE2: Beam spline (1D 보간)
  - TRIM: 트림 조건 (Mach, q, ANGLEA, 제어면 등)
  - AESTAT: 강체 공력 변수 (ANGLEA, SIDES, URDD3 등)
  - AESURF: 제어면 정의
  - FLFACT: 밀도/Mach/속도 리스트
  - MKAERO1: Mach-k 조합
- `nastaero/bdf/bulk_data.py` 업데이트 (디스패치 추가)

### Step 3: DLM 핵심 모듈
- `nastaero/aero/__init__.py`
- `nastaero/aero/panel.py` (~150줄)
  - `AeroBox` 데이터클래스 (corner, control point, downwash point, normal, area)
  - `generate_panel_mesh(caero1)` → List[AeroBox] (nspan×nchord 박스 생성)
  - DLM 규약: control point 1/4c, downwash point 3/4c
- `nastaero/aero/dlm.py` (~250줄)
  - `dlm_kernel(k, r, e, u)` → complex (Desmarais 5항 근사)
  - `compute_box_to_box_aic(send, recv, k, mach)` → complex
  - `build_aic_matrix(boxes, k, mach)` → np.ndarray [n×n]
  - Prandtl-Glauert 압축성 보정 (β = √(1-M²))

### Step 4: 구조-공력 스플라인
- `nastaero/aero/spline.py` (~250줄)
  - `build_infinite_plate_spline(struct_nodes, aero_boxes, spline1)` → (G_ka, G_kg)
    - Green 함수: G(r) = r²(ln(r)-1)/(8π)
    - 다항식 보강 (강체 운동 정확 재현)
  - `build_beam_spline(struct_nodes, aero_boxes, spline2)` → (G_ka, G_kg)
  - `AeroStructureInterface` 클래스: 공력 메시 생성 + 스플라인 조립 + AIC 계산

### Step 5: SOL 144 솔버
- `nastaero/solvers/sol144.py` (~300줄)
  - `solve_static_aero_trim(bdf_model)` → ResultData
  - 연립방정식:
    ```
    [K_aa + q·Q_aa  |  q·Q_ax] [u_a]   [P_a    ]
    [    D_a        |   D_x  ] [x  ] = [rhs_trim]
    ```
    - Q_aa = G_ka^T · Q_jj · G_ka (구조에 대한 공력 영향)
    - Q_ax = G_ka^T · Q_jj · dw/dx (트림 변수의 공력 영향)
    - D_a, D_x: 힘/모멘트 평형 구속 조건
  - `build_trim_equations()`: 트림 구속 조건 조립
  - `compute_anglea_normalwash()`: 받음각에 의한 normalwash
  - `compute_aero_forces_moments()`: 공력 하중 적분

### Step 6: 출력 확장
- `nastaero/output/result_data.py` 확장: trim_variables, aero_pressures, aero_forces/moments
- `nastaero/output/f06_writer.py` 확장: 트림 결과, 공력 하중 출력
- `nastaero/__main__.py` 확장: SOL 144 지원

### Step 7: 검증
- `tests/validation/goland_wing/goland_static.bdf`
  - Goland wing: L=6.096m, c=1.8288m, 탄성축 33% chord
  - 구조: 10 CBAR, 공력: 8×2 DLM 패널, SPLINE1 연결
  - TRIM: M=0.3, level flight (1g)
  - 기대 결과: ANGLEA 2~6도, tip deflection < 1m
- `tests/test_aero_cards.py`: 카드 파싱 테스트
- `tests/test_aero/test_panel.py`: 패널 메시 생성
- `tests/test_aero/test_dlm.py`: AIC 행렬 대칭성, 평판 양력 기울기
- `tests/test_aero/test_spline.py`: 강체 운동 재현, 보간 정확도
- `tests/test_sol144.py`: Goland wing 트림 검증

### 핵심 파일 목록 (생성/수정)
```
신규 생성:
  nastaero/bdf/cards/aero.py       # 에어로 BDF 카드 파서
  nastaero/bdf/cards/sets.py       # SET1 카드
  nastaero/aero/__init__.py        # 패키지
  nastaero/aero/panel.py           # 공력 패널 메시
  nastaero/aero/dlm.py             # DLM 커널 + AIC
  nastaero/aero/spline.py          # 구조-공력 스플라인
  nastaero/solvers/sol144.py       # SOL 144 솔버
  tests/test_aero_cards.py         # 카드 파싱 테스트
  tests/test_aero/                 # 공력 모듈 테스트
  tests/validation/goland_wing/    # Goland wing BDF

기존 수정:
  nastaero/bdf/cards/rbe.py        # RBE3 추가
  nastaero/bdf/bulk_data.py        # 카드 디스패치 추가
  nastaero/output/result_data.py   # 트림 결과 필드
  nastaero/output/f06_writer.py    # 트림 출력
  nastaero/__main__.py             # SOL 144 CLI
```

### 검증 방법
```bash
cd D:\git\NastAero\solver
pytest tests/ -v                    # 기존 63개 + 신규 ~15개 = 78+ 테스트
python -m nastaero tests/validation/goland_wing/goland_static.bdf
```
