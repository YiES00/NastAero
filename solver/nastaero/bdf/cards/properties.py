"""Property card parsers."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple
import numpy as np
from ..field_parser import nastran_int, nastran_float

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

def _compute_tube(dims):
    """TUBE: R_outer, t"""
    Ro = dims[0]; t = dims[1]
    Ri = Ro - t
    if Ri < 0: Ri = 0
    A = math.pi * (Ro**2 - Ri**2)
    Ix = Iy = math.pi * (Ro**4 - Ri**4) / 4.0
    J = math.pi * (Ro**4 - Ri**4) / 2.0
    return A, Ix, Iy, J

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

def _compute_i_section(dims):
    """I: w_top, w_bot, h, t_top, t_bot, t_web
    I-section with possibly different top/bottom flanges.
    """
    wt = dims[0]; wb = dims[1]; h = dims[2]
    tt = dims[3]; tb = dims[4]; tw = dims[5]
    hw = h - tt - tb  # web height
    # Areas
    A_top = wt * tt; A_bot = wb * tb; A_web = hw * tw
    A = A_top + A_bot + A_web
    # Centroid from bottom
    y_top = h - tt/2; y_bot = tb/2; y_web = tb + hw/2
    if A > 0:
        yc = (A_top*y_top + A_bot*y_bot + A_web*y_web) / A
    else:
        yc = h/2
    # Ix about centroid (parallel axis theorem)
    Ix = (wt*tt**3/12 + A_top*(y_top-yc)**2 +
          wb*tb**3/12 + A_bot*(y_bot-yc)**2 +
          tw*hw**3/12 + A_web*(y_web-yc)**2)
    Iy = tt*wt**3/12 + tb*wb**3/12 + hw*tw**3/12
    # Approximate torsion constant
    J = (wt*tt**3 + wb*tb**3 + hw*tw**3) / 3.0
    return A, Ix, Iy, J

def _compute_i1_section(dims):
    """I1: w_bot, w_top, h_total, t_web
    I1 section with equal flanges (simplified I).
    """
    wb = dims[0]; wt = dims[1]; h = dims[2]; tw = dims[3]
    # Approximate flange thickness from geometry
    tf = tw  # Assume flange thickness = web thickness if not specified
    return _compute_i_section([wt, wb, h, tf, tf, tw])

def _compute_chan1(dims):
    """CHAN1: w_flange, t_flange, h, t_web
    Channel section (open, no torsional stiffness).
    """
    wf = dims[0]; tf = dims[1]; h = dims[2]; tw = dims[3]
    hw = h - 2*tf
    A = 2*wf*tf + hw*tw
    yc = h/2  # symmetric about y
    Ix = 2*(wf*tf**3/12 + wf*tf*(h/2-tf/2)**2) + tw*hw**3/12
    Iy = 2*tf*wf**3/12 + hw*tw**3/12
    J = (2*wf*tf**3 + hw*tw**3) / 3.0
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
    'ROD': _compute_rod, 'TUBE': _compute_tube, 'BAR': _compute_bar,
    'BOX': _compute_box, 'I': _compute_i_section, 'I1': _compute_i1_section,
    'CHAN1': _compute_chan1, 'L': _compute_l_section, 'T': _compute_t_section,
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
        """Compute A, I1, I2, J from cross-section type and dimensions."""
        compute_fn = _SECTION_COMPUTE.get(self.type_name)
        if compute_fn and self.dims:
            try:
                self.A, self.I1, self.I2, self.J = compute_fn(self.dims)
            except (IndexError, ZeroDivisionError, ValueError):
                pass  # Leave as zeros


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

    def compute_section(self):
        """Compute A, I1, I2, J from cross-section type and dimensions."""
        compute_fn = _SECTION_COMPUTE.get(self.type_name)
        if compute_fn and self.dims:
            try:
                self.A, self.I1, self.I2, self.J = compute_fn(self.dims)
            except (IndexError, ZeroDivisionError, ValueError):
                pass


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
