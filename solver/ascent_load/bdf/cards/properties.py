"""Property card parsers."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple
import logging
import numpy as np
from ..field_parser import nastran_int, nastran_float

logger = logging.getLogger(__name__)

@dataclass
class PBAR:
    pid: int = 0
    mid: int = 0
    A: float = 0.0
    I1: float = 0.0
    I2: float = 0.0
    J: float = 0.0
    nsm: float = 0.0
    c1: float = 0.0; c2: float = 0.0; d1: float = 0.0; d2: float = 0.0
    e1: float = 0.0; e2: float = 0.0; f1: float = 0.0; f2: float = 0.0
    material_ref: Optional[Any] = None

    @classmethod
    def from_fields(cls, fields: List[str]) -> PBAR:
        p = cls()
        p.pid = nastran_int(fields[1]); p.mid = nastran_int(fields[2])
        p.A = nastran_float(fields[3]); p.I1 = nastran_float(fields[4])
        p.I2 = nastran_float(fields[5]); p.J = nastran_float(fields[6])
        p.nsm = nastran_float(fields[7]) if len(fields) > 7 else 0.0
        # 연속행 필드 9~16 = C1 C2 D1 D2 E1 E2 F1 F2 (응력 회수점).
        # 종전에는 파싱하지 않아 응력 회수가 c ~ sqrt(I/A) 근사를
        # 썼고, 이는 원형 단면에서도 최외곽 거리의 절반이다.
        def f(i):
            return nastran_float(fields[i]) if len(fields) > i else 0.0
        p.c1, p.c2 = f(9), f(10)
        p.d1, p.d2 = f(11), f(12)
        p.e1, p.e2 = f(13), f(14)
        p.f1, p.f2 = f(15), f(16)
        return p

@dataclass
class PBEAM:
    """CBEAM 표 형식 단면 물성 (A단 스테이션 기준).

    MSC 필드 순서는 PBAR와 다르다: PID MID A I1 I2 I12 J NSM으로
    I12가 J보다 앞에 온다. PBAR 순서로 읽으면 J 자리에 I12가
    들어가 비틀림 강성이 통째로 어긋난다.

    PBEAM은 스테이션별 단면을 표로 줄 수 있으나 본 솔버의 보
    요소는 균일 단면이므로 A단 스테이션 값을 사용한다. 스테이션이
    여럿이면 경고한다.
    """
    pid: int = 0
    mid: int = 0
    A: float = 0.0
    I1: float = 0.0
    I2: float = 0.0
    I12: float = 0.0
    J: float = 0.0
    nsm: float = 0.0
    material_ref: Optional[Any] = None

    @classmethod
    def from_fields(cls, fields: List[str]) -> PBEAM:
        def f(i):
            return nastran_float(fields[i]) if len(fields) > i else 0.0
        p = cls()
        p.pid = nastran_int(fields[1])
        p.mid = nastran_int(fields[2])
        p.A = f(3)
        p.I1 = f(4)
        p.I2 = f(5)
        p.I12 = f(6)
        p.J = f(7)
        p.nsm = f(8)
        # 연속행에 추가 스테이션이 있으면 A단만 쓴다는 사실을 알린다
        if len(fields) > 17 and any(x.strip() for x in fields[17:]):
            logger.warning(
                "PBEAM %d: 스테이션이 여럿이지만 A단 단면만 사용한다 "
                "(균일 단면 보 요소)", p.pid)
        if p.A <= 0.0:
            logger.warning("PBEAM %d: 단면적 %.6g <= 0", p.pid, p.A)
        return p


@dataclass
class PROD:
    pid: int = 0; mid: int = 0; A: float = 0.0; J: float = 0.0
    c: float = 0.0; nsm: float = 0.0; material_ref: Optional[Any] = None
    @classmethod
    def from_fields(cls, fields: List[str]) -> PROD:
        p = cls(); p.pid = nastran_int(fields[1]); p.mid = nastran_int(fields[2])
        p.A = nastran_float(fields[3])
        p.J = nastran_float(fields[4]) if len(fields) > 4 else 0.0
        return p

@dataclass
class PSHELL:
    pid: int = 0; mid: int = 0; t: float = 0.0; mid2: int = 0
    ratio_12it3: float = 1.0; mid3: int = 0; ts_t: float = 0.833333; nsm: float = 0.0
    material_ref: Optional[Any] = None
    @classmethod
    def from_fields(cls, fields: List[str]) -> PSHELL:
        p = cls(); p.pid = nastran_int(fields[1]); p.mid = nastran_int(fields[2])
        p.t = nastran_float(fields[3])
        p.mid2 = nastran_int(fields[4]) if len(fields) > 4 and fields[4].strip() else p.mid
        p.ratio_12it3 = nastran_float(fields[5], default=1.0) if len(fields) > 5 else 1.0
        p.mid3 = nastran_int(fields[6]) if len(fields) > 6 and fields[6].strip() else 0
        p.ts_t = nastran_float(fields[7], default=0.833333) if len(fields) > 7 else 0.833333
        p.nsm = nastran_float(fields[8]) if len(fields) > 8 else 0.0
        return p

@dataclass
class PSOLID:
    pid: int = 0; mid: int = 0; cordm: int = 0; material_ref: Optional[Any] = None
    @classmethod
    def from_fields(cls, fields: List[str]) -> PSOLID:
        p = cls(); p.pid = nastran_int(fields[1]); p.mid = nastran_int(fields[2])
        p.cordm = nastran_int(fields[3]) if len(fields) > 3 else 0
        return p


@dataclass
class PCOMP:
    """Composite laminate shell property.

    PCOMP PID  Z0  NSM  SB  FT  TREF  GE  LAM
          MID1 T1  THETA1 SOUT1  MID2 T2  THETA2 SOUT2
          ...

    When LAM='SYM', the ply stack is mirrored symmetrically.
    """
    pid: int = 0
    z0: float = 0.0        # Offset from reference plane (default = -t/2)
    nsm: float = 0.0       # Non-structural mass per area
    sb: float = 0.0        # Allowable interlaminar shear stress
    ft: str = ""            # Failure theory (HILL, HOFF, TSAI, STRN)
    tref: float = 0.0      # Reference temperature
    ge: float = 0.0        # Structural damping
    lam: str = ""           # Laminate option: "", "SYM", "MEM", "BEND"
    plies: List[Tuple[int, float, float, str]] = field(default_factory=list)
    # (mid, thickness, theta_deg, sout)
    material_ref: Optional[Any] = None  # Not used directly; plies have own mats
    ply_materials: List[Any] = field(default_factory=list)
    # Cached equivalent properties
    _eq_E: float = 0.0
    _eq_nu: float = 0.0
    _eq_t: float = 0.0
    _eq_rho: float = 0.0
    # Compatibility with PSHELL interface
    mid: int = 0
    t: float = 0.0

    @classmethod
    def from_fields(cls, fields: List[str]) -> PCOMP:
        p = cls()
        p.pid = nastran_int(fields[1])
        p.z0 = nastran_float(fields[2]) if len(fields) > 2 and fields[2].strip() else 0.0
        p.nsm = nastran_float(fields[3]) if len(fields) > 3 and fields[3].strip() else 0.0
        p.sb = nastran_float(fields[4]) if len(fields) > 4 and fields[4].strip() else 0.0
        p.ft = fields[5].strip() if len(fields) > 5 and fields[5].strip() else ""
        p.tref = nastran_float(fields[6]) if len(fields) > 6 and fields[6].strip() else 0.0
        p.ge = nastran_float(fields[7]) if len(fields) > 7 and fields[7].strip() else 0.0
        p.lam = fields[8].strip().upper() if len(fields) > 8 and fields[8].strip() else ""

        # Parse ply data: groups of 4 fields (MID, T, THETA, SOUT) starting from field 9
        i = 9
        while i + 1 < len(fields):
            mid_s = fields[i].strip() if i < len(fields) else ""
            if not mid_s:
                i += 4; continue
            try:
                mid = int(mid_s)
            except (ValueError, TypeError):
                i += 4; continue
            t = nastran_float(fields[i+1]) if i+1 < len(fields) else 0.0
            theta = nastran_float(fields[i+2]) if i+2 < len(fields) else 0.0
            sout = fields[i+3].strip() if i+3 < len(fields) else "NO"
            p.plies.append((mid, t, theta, sout))
            i += 4

        # Apply SYM laminate option: mirror plies
        if p.lam == "SYM" and p.plies:
            p.plies = p.plies + list(reversed(p.plies))

        # Compute total thickness
        p.t = sum(ply[1] for ply in p.plies)
        if p.plies:
            p.mid = p.plies[0][0]  # First ply material for cross-reference

        return p

    def equivalent_isotropic(self, materials=None):
        """Compute smeared equivalent isotropic properties.

        Returns (E, nu, t, rho) for use with existing shell element formulation.
        """
        if self._eq_E > 0:
            return self._eq_E, self._eq_nu, self._eq_t, self._eq_rho

        total_t = sum(ply[1] for ply in self.plies)
        if total_t < 1e-30:
            return 0.0, 0.3, 0.0, 0.0

        # If materials dict provided, compute proper CLT A-matrix
        if materials:
            A = np.zeros((3, 3))
            total_rho_t = 0.0
            z_bot = -total_t / 2.0
            for mid, t, theta, _ in self.plies:
                if mid in materials:
                    mat = materials[mid]
                    if hasattr(mat, 'plane_stress_Q'):
                        Q = mat.plane_stress_Q()
                    else:
                        E = mat.E; nu = mat.nu
                        denom = 1.0 - nu**2
                        if abs(denom) < 1e-30: denom = 1.0
                        G = mat.G if mat.G > 0 else E / (2*(1+nu))
                        Q = np.array([[E/denom, nu*E/denom, 0],
                                      [nu*E/denom, E/denom, 0],
                                      [0, 0, G]])
                    # Rotate Q by theta
                    Qbar = _rotate_Q(Q, theta)
                    A += Qbar * t
                    total_rho_t += mat.rho * t
                z_bot += t

            E_eq = A[0, 0] / total_t if total_t > 0 else 0.0
            nu_eq = A[0, 1] / A[0, 0] if abs(A[0, 0]) > 1e-30 else 0.3
            rho_eq = total_rho_t / total_t if total_t > 0 else 0.0
        else:
            E_eq = 0.0; nu_eq = 0.3; rho_eq = 0.0
            if self.ply_materials:
                E_sum = 0.0; rho_sum = 0.0
                for i, (mid, t, theta, _) in enumerate(self.plies):
                    if i < len(self.ply_materials) and self.ply_materials[i]:
                        m = self.ply_materials[i]
                        E_sum += (m.E if hasattr(m, 'E') else m.E1) * t
                        rho_sum += m.rho * t
                E_eq = E_sum / total_t if total_t > 0 else 0.0
                rho_eq = rho_sum / total_t if total_t > 0 else 0.0

        self._eq_E = E_eq
        self._eq_nu = nu_eq
        self._eq_t = total_t
        self._eq_rho = rho_eq
        return E_eq, nu_eq, total_t, rho_eq


def _rotate_Q(Q, theta_deg):
    """Rotate a 2D stiffness matrix Q by angle theta (degrees)."""
    theta = np.radians(theta_deg)
    c = np.cos(theta); s = np.sin(theta)
    T = np.array([[c*c, s*s, 2*c*s],
                  [s*s, c*c, -2*c*s],
                  [-c*s, c*s, c*c-s*s]])
    T_inv = np.array([[c*c, s*s, -2*c*s],
                      [s*s, c*c, 2*c*s],
                      [c*s, -c*s, c*c-s*s]])
    return T_inv @ Q @ T_inv.T


# =====================================================================
# Section shape computation helpers for PBARL / PBEAML
# =====================================================================

def _compute_rod(dims):
    """ROD: R"""
    R = dims[0]
    A = math.pi * R**2
    Ix = Iy = math.pi * R**4 / 4.0
    J = math.pi * R**4 / 2.0
    return A, Ix, Iy, J

def _tube_from_radii(Ro, Ri):
    Ri = min(max(Ri, 0.0), Ro)
    A = math.pi * (Ro**2 - Ri**2)
    Ix = Iy = math.pi * (Ro**4 - Ri**4) / 4.0
    J = math.pi * (Ro**4 - Ri**4) / 2.0
    return A, Ix, Iy, J

def _compute_tube(dims):
    """TUBE: DIM1 = 외경 반지름, DIM2 = 내경 반지름 (MSC 규약).

    두께가 아니라 내반지름이다. 두께 변형은 TUBE2가 담당한다.
    """
    return _tube_from_radii(dims[0], dims[1])

def _compute_tube2(dims):
    """TUBE2: DIM1 = 외경 반지름, DIM2 = 벽 두께."""
    return _tube_from_radii(dims[0], dims[0] - dims[1])

def _compute_bar(dims):
    """BAR: width, height"""
    w = dims[0]; h = dims[1]
    A = w * h
    Ix = w * h**3 / 12.0  # about centroid
    Iy = h * w**3 / 12.0
    # Torsion constant for rectangle
    a = max(w, h) / 2.0; b = min(w, h) / 2.0
    J = a * b**3 * (16.0/3.0 - 3.36 * b/a * (1.0 - b**4/(12.0*a**4)))
    return A, Ix, Iy, J

def _compute_box(dims):
    """BOX: w, h, t1, t2 [, t3, t4]
    t1=top, t2=bottom, t3=left(default=t1), t4=right(default=t2)
    """
    w = dims[0]; h = dims[1]; t1 = dims[2]; t2 = dims[3]
    t3 = dims[4] if len(dims) > 4 else t1
    t4 = dims[5] if len(dims) > 5 else t2
    # Approximate thin-walled box
    A = w*h - (w-t3-t4)*(h-t1-t2)
    if A < 0: A = w*h
    Ix = w*h**3/12.0 - (w-t3-t4)*max(h-t1-t2,0)**3/12.0
    Iy = h*w**3/12.0 - max(h-t1-t2,0)*(w-t3-t4)**3/12.0
    # Bredt formula for thin-walled closed section
    t_avg = (t1+t2+t3+t4)/4.0
    Am = (w-t_avg)*(h-t_avg)
    perimeter = 2*((w-t_avg) + (h-t_avg))
    J = 4*Am**2*t_avg / perimeter if perimeter > 0 else 0.0
    return A, Ix, Iy, J

def _i_section(h, wb, wt, tw, tb, tt):
    """상/하 플랜지가 다를 수 있는 I 단면의 A, Ix, Iy, J."""
    hw = max(h - tt - tb, 0.0)
    A_top = wt * tt; A_bot = wb * tb; A_web = hw * tw
    A = A_top + A_bot + A_web
    y_top = h - tt / 2.0; y_bot = tb / 2.0; y_web = tb + hw / 2.0
    yc = ((A_top * y_top + A_bot * y_bot + A_web * y_web) / A
          if A > 0 else h / 2.0)
    Ix = (wt * tt**3 / 12 + A_top * (y_top - yc)**2 +
          wb * tb**3 / 12 + A_bot * (y_bot - yc)**2 +
          tw * hw**3 / 12 + A_web * (y_web - yc)**2)
    Iy = tt * wt**3 / 12 + tb * wb**3 / 12 + hw * tw**3 / 12
    J = (wt * tt**3 + wb * tb**3 + hw * tw**3) / 3.0
    return A, Ix, Iy, J

def _compute_i_section(dims):
    """I: DIM1=전체 깊이, DIM2=하부 플랜지 폭, DIM3=상부 플랜지 폭,
    DIM4=웨브 두께, DIM5=하부 플랜지 두께, DIM6=상부 플랜지 두께.
    """
    h, wb, wt, tw, tb, tt = dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]
    return _i_section(h, wb, wt, tw, tb, tt)

def _compute_i1_section(dims):
    """I1: DIM1=플랜지 폭 - 웨브 두께, DIM2=웨브 두께,
    DIM3=플랜지 사이 순 웨브 높이, DIM4=전체 깊이. 상하 플랜지 동일.
    """
    tw = dims[1]; bf = dims[0] + tw; hw = dims[2]; h = dims[3]
    tf = max((h - hw) / 2.0, 0.0)
    return _i_section(h, bf, bf, tw, tf, tf)

def _channel(bf, tf, h, tw):
    """웨브가 수직인 채널(ㄷ) 단면. bf=플랜지 전체 폭, h=전체 깊이."""
    hw = max(h - 2.0 * tf, 0.0)
    A_f = bf * tf; A_w = hw * tw
    A = 2.0 * A_f + A_w
    # 깊이 방향(강축) 관성: 상하 대칭이므로 도심은 중앙
    Ix = 2.0 * (bf * tf**3 / 12 + A_f * ((h - tf) / 2.0)**2) + tw * hw**3 / 12
    # 폭 방향(약축) 관성: 웨브가 한쪽에 몰려 도심이 치우친다
    A_tot = A if A > 0 else 1.0
    xc = (2.0 * A_f * bf / 2.0 + A_w * tw / 2.0) / A_tot
    Iy = (2.0 * (tf * bf**3 / 12 + A_f * (bf / 2.0 - xc)**2) +
          hw * tw**3 / 12 + A_w * (tw / 2.0 - xc)**2)
    J = (2.0 * bf * tf**3 + hw * tw**3) / 3.0
    return A, Ix, Iy, J

def _compute_chan(dims):
    """CHAN: DIM1=플랜지 폭, DIM2=전체 깊이, DIM3=웨브 두께,
    DIM4=플랜지 두께.
    """
    return _channel(dims[0], dims[3], dims[1], dims[2])

def _compute_chan1(dims):
    """CHAN1: DIM1=웨브 바깥 플랜지 폭, DIM2=웨브 두께,
    DIM3=플랜지 사이 순 웨브 높이, DIM4=전체 깊이.
    """
    tw = dims[1]; bf = dims[0] + tw; hw = dims[2]; h = dims[3]
    tf = max((h - hw) / 2.0, 0.0)
    return _channel(bf, tf, h, tw)

def _compute_chan2(dims):
    """CHAN2: DIM1=플랜지 두께, DIM2=웨브 두께, DIM3=전체 깊이,
    DIM4=전체 폭. 웨브가 수평인 ㄴ자 반전(U) 배치다.
    """
    tf, tw, h, w = dims[0], dims[1], dims[2], dims[3]
    hf = max(h - tw, 0.0)
    A_web = w * tw; A_f = tf * hf
    A = A_web + 2.0 * A_f
    A_tot = A if A > 0 else 1.0
    yc = (A_web * tw / 2.0 + 2.0 * A_f * (tw + hf / 2.0)) / A_tot
    Ix = (w * tw**3 / 12 + A_web * (tw / 2.0 - yc)**2 +
          2.0 * (tf * hf**3 / 12 + A_f * (tw + hf / 2.0 - yc)**2))
    Iy = (tw * w**3 / 12 +
          2.0 * (hf * tf**3 / 12 + A_f * ((w - tf) / 2.0)**2))
    J = (w * tw**3 + 2.0 * hf * tf**3) / 3.0
    return A, Ix, Iy, J

def _compute_z_section(dims):
    """Z: DIM1=플랜지 폭, DIM2=웨브 두께, DIM3=순 웨브 높이,
    DIM4=전체 깊이. 플랜지가 서로 반대로 뻗는 점만 채널과 다르므로
    면적과 강축 관성은 같고 약축 관성은 단면 자체 축 기준으로 잡는다.
    """
    tw = dims[1]; bf = dims[0] + tw; hw = dims[2]; h = dims[3]
    tf = max((h - hw) / 2.0, 0.0)
    hw_c = max(h - 2.0 * tf, 0.0)
    A_f = bf * tf; A = 2.0 * A_f + hw_c * tw
    Ix = (2.0 * (bf * tf**3 / 12 + A_f * ((h - tf) / 2.0)**2) +
          tw * hw_c**3 / 12)
    # Z는 도심이 웨브 중앙이라 약축 관성이 채널과 다르다
    Iy = 2.0 * (tf * bf**3 / 12 + A_f * ((bf - tw) / 2.0)**2) + hw_c * tw**3 / 12
    J = (2.0 * bf * tf**3 + hw_c * tw**3) / 3.0
    return A, Ix, Iy, J

def _compute_hat(dims):
    """HAT: DIM1=전체 깊이, DIM2=벽 두께, DIM3=한쪽 브림 폭,
    DIM4=상부 폭. 균일 두께 박벽 조립체로 계산한다.
    """
    h, t, wb, wt = dims[0], dims[1], dims[2], dims[3]
    hw = max(h - t, 0.0)
    A_top = wt * t; A_web = hw * t; A_brim = wb * t
    A = A_top + 2.0 * A_web + 2.0 * A_brim
    A_tot = A if A > 0 else 1.0
    y_top = h - t / 2.0; y_web = t + hw / 2.0; y_brim = t / 2.0
    yc = (A_top * y_top + 2.0 * A_web * y_web + 2.0 * A_brim * y_brim) / A_tot
    Ix = (wt * t**3 / 12 + A_top * (y_top - yc)**2 +
          2.0 * (t * hw**3 / 12 + A_web * (y_web - yc)**2) +
          2.0 * (wb * t**3 / 12 + A_brim * (y_brim - yc)**2))
    half = wt / 2.0
    Iy = (t * wt**3 / 12 +
          2.0 * (hw * t**3 / 12 + A_web * half**2) +
          2.0 * (t * wb**3 / 12 + A_brim * (half + wb / 2.0)**2))
    J = (wt * t**3 + 2.0 * hw * t**3 + 2.0 * wb * t**3) / 3.0
    return A, Ix, Iy, J

def _compute_l_section(dims):
    """L: w1, w2, t1, t2  (angle section)"""
    w1 = dims[0]; w2 = dims[1]; t1 = dims[2]; t2 = dims[3]
    A1 = w1 * t1; A2 = (w2-t1) * t2
    A = A1 + A2
    xc = (A1*t1/2 + A2*(t1+(w2-t1)/2)) / A if A > 0 else 0
    yc = (A1*w1/2 + A2*t2/2) / A if A > 0 else 0
    Ix = t1*w1**3/12 + A1*(w1/2-yc)**2 + (w2-t1)*t2**3/12 + A2*(t2/2-yc)**2
    Iy = w1*t1**3/12 + A1*(t1/2-xc)**2 + t2*(w2-t1)**3/12 + A2*((t1+(w2-t1)/2)-xc)**2
    J = (w1*t1**3 + (w2-t1)*t2**3) / 3.0
    return A, Ix, Iy, J

def _compute_t_section(dims):
    """T: w_flange, h, t_flange, t_web"""
    wf = dims[0]; h = dims[1]; tf = dims[2]; tw = dims[3]
    hw = h - tf
    A_f = wf * tf; A_w = hw * tw
    A = A_f + A_w
    yc = (A_f*(h-tf/2) + A_w*hw/2) / A if A > 0 else h/2
    Ix = wf*tf**3/12 + A_f*(h-tf/2-yc)**2 + tw*hw**3/12 + A_w*(hw/2-yc)**2
    Iy = tf*wf**3/12 + hw*tw**3/12
    J = (wf*tf**3 + hw*tw**3) / 3.0
    return A, Ix, Iy, J

_SECTION_COMPUTE = {
    'ROD': _compute_rod, 'TUBE': _compute_tube, 'TUBE2': _compute_tube2,
    'BAR': _compute_bar, 'BOX': _compute_box,
    'I': _compute_i_section, 'I1': _compute_i1_section,
    'CHAN': _compute_chan, 'CHAN1': _compute_chan1, 'CHAN2': _compute_chan2,
    'Z': _compute_z_section, 'HAT': _compute_hat,
    'L': _compute_l_section, 'T': _compute_t_section,
}

# 타입별 DIM 개수 — 그 뒤 필드는 NSM이다 (dims에 삼키면 단면이 깨진다)
_SECTION_NDIM = {
    'ROD': 1, 'TUBE': 2, 'TUBE2': 2, 'BAR': 2, 'BOX': 6,
    'I': 6, 'I1': 4, 'CHAN': 4, 'CHAN1': 4, 'CHAN2': 4,
    'Z': 4, 'HAT': 4, 'L': 4, 'T': 4,
}


@dataclass
class PBARL:
    """Parametric bar cross-section property.

    PBARL  PID  MID  GROUP  TYPE
           DIM1 DIM2 DIM3 ...  NSM
    """
    pid: int = 0
    mid: int = 0
    group: str = "MSCBML0"
    type_name: str = ""
    dims: List[float] = field(default_factory=list)
    nsm: float = 0.0
    # Computed section properties (PBAR-compatible)
    A: float = 0.0
    I1: float = 0.0
    I2: float = 0.0
    J: float = 0.0
    material_ref: Optional[Any] = None

    @classmethod
    def from_fields(cls, fields: List[str]) -> PBARL:
        p = cls()
        p.pid = nastran_int(fields[1])
        p.mid = nastran_int(fields[2])
        p.group = fields[3].strip() if len(fields) > 3 and fields[3].strip() else "MSCBML0"
        p.type_name = fields[4].strip().upper() if len(fields) > 4 else ""
        # Parse dimensions from continuation (field 9 onwards)
        for f in fields[9:]:
            s = f.strip()
            if s:
                try:
                    p.dims.append(nastran_float(s))
                except (ValueError, TypeError):
                    break
        p.compute_section()
        return p

    def compute_section(self):
        """단면 타입과 치수로 A, I1, I2, J를 계산한다.

        타입별 DIM 개수를 넘는 값은 NSM이므로 분리한다. 계산에
        실패하거나 면적이 0 이하이면 그 요소는 강성이 없는 채로
        조립되므로 조용히 넘기지 않고 경고한다.
        """
        ndim = _SECTION_NDIM.get(self.type_name)
        if ndim is not None and len(self.dims) > ndim:
            self.nsm = self.dims[ndim]
            self.dims = self.dims[:ndim]

        compute_fn = _SECTION_COMPUTE.get(self.type_name)
        if compute_fn is None:
            logger.warning(
                "%s %d: 단면 타입 '%s'는 미지원 — 강성 0으로 조립된다",
                type(self).__name__, self.pid, self.type_name)
            return
        if not self.dims:
            logger.warning("%s %d: 단면 치수 없음 (타입 %s)",
                           type(self).__name__, self.pid, self.type_name)
            return
        try:
            self.A, self.I1, self.I2, self.J = compute_fn(self.dims)
        except (IndexError, ZeroDivisionError, ValueError) as exc:
            logger.warning("%s %d (%s): 단면 계산 실패 (%s) — 강성 0",
                           type(self).__name__, self.pid, self.type_name, exc)
            return
        if self.A <= 0.0:
            logger.warning(
                "%s %d (%s): 단면적 %.6g <= 0 — 치수 %s 확인 필요",
                type(self).__name__, self.pid, self.type_name,
                self.A, self.dims)


@dataclass
class PBEAML:
    """Parametric beam cross-section property (same format as PBARL).

    PBEAML PID  MID  GROUP  TYPE
           DIM1 DIM2 DIM3 ...  NSM
    """
    pid: int = 0
    mid: int = 0
    group: str = "MSCBML0"
    type_name: str = ""
    dims: List[float] = field(default_factory=list)
    nsm: float = 0.0
    A: float = 0.0
    I1: float = 0.0
    I2: float = 0.0
    J: float = 0.0
    material_ref: Optional[Any] = None

    @classmethod
    def from_fields(cls, fields: List[str]) -> PBEAML:
        p = cls()
        p.pid = nastran_int(fields[1])
        p.mid = nastran_int(fields[2])
        p.group = fields[3].strip() if len(fields) > 3 and fields[3].strip() else "MSCBML0"
        p.type_name = fields[4].strip().upper() if len(fields) > 4 else ""
        for f in fields[9:]:
            s = f.strip()
            if s:
                try:
                    p.dims.append(nastran_float(s))
                except (ValueError, TypeError):
                    break
        p.compute_section()
        return p

    compute_section = PBARL.compute_section


@dataclass
class PELAS:
    """Scalar spring property.

    PELAS  PID  K  GE  S
    """
    pid: int = 0
    k: float = 0.0   # Spring stiffness
    ge: float = 0.0   # Damping coefficient
    s: float = 0.0    # Stress coefficient
    material_ref: Optional[Any] = None  # Not used, for compatibility

    @classmethod
    def from_fields(cls, fields: List[str]) -> PELAS:
        p = cls()
        p.pid = nastran_int(fields[1])
        p.k = nastran_float(fields[2])
        p.ge = nastran_float(fields[3]) if len(fields) > 3 else 0.0
        p.s = nastran_float(fields[4]) if len(fields) > 4 else 0.0
        return p
