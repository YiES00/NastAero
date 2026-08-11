"""Airfoil camber models for DLM/VLM normalwash correction.

Provides camber line slope (dz_c/dx) computation for NACA 4-digit and
5-digit airfoils.  The camber normalwash correction is added to the
VLM boundary condition so that flat-plate panels behave as if they
have the specified airfoil camber.

VLM boundary condition (flat plate):  w/V = -alpha
With camber correction:               w/V = -alpha + dz_c/dx

References
----------
- Abbott & von Doenhoff, "Theory of Wing Sections", Dover, 1959
- NASA TN D-7428: "Computer Programs for NACA Airfoil Sections"
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .panel import AeroBox


@dataclass
class AirfoilCamber:
    """Airfoil camber line definition.

    Attributes
    ----------
    name : str
        Airfoil designation (e.g., 'NACA4415', 'NACA23015').
    max_camber : float
        Maximum camber as fraction of chord (m in NACA notation).
    camber_pos : float
        Chordwise position of max camber as fraction (p in NACA notation).
    series : str
        '4digit' or '5digit'.
    cl_design : float
        Design lift coefficient (5-digit only).
    k1 : float
        Camber line constant (5-digit only).
    """
    name: str = "NACA0012"
    max_camber: float = 0.0
    camber_pos: float = 0.0
    series: str = "4digit"
    cl_design: float = 0.0
    k1: float = 0.0

    def camber_slope(self, x_frac: float) -> float:
        """Compute camber line slope dz_c/dx at normalized position x/c.

        Parameters
        ----------
        x_frac : float
            Chordwise position as fraction of chord (0 = LE, 1 = TE).

        Returns
        -------
        float
            Camber line slope dz_c/d(x/c).
        """
        x = np.clip(x_frac, 0.0, 1.0)

        if self.max_camber < 1e-8:
            return 0.0

        if self.series == "4digit":
            return self._naca4_slope(x)
        elif self.series == "5digit":
            return self._naca5_slope(x)
        return 0.0

    def _naca4_slope(self, x: float) -> float:
        """NACA 4-digit camber line slope.

        Camber line:
            x < p:  z_c/c = (m/p^2)(2p*x - x^2)
            x >= p: z_c/c = (m/(1-p)^2)((1-2p) + 2p*x - x^2)

        Slope:
            x < p:  dz/dx = (2m/p^2)(p - x)
            x >= p: dz/dx = (2m/(1-p)^2)(p - x)
        """
        m = self.max_camber
        p = self.camber_pos
        if p < 1e-8 or p > 1.0 - 1e-8:
            return 0.0

        if x < p:
            return (2.0 * m / (p * p)) * (p - x)
        else:
            return (2.0 * m / ((1.0 - p) ** 2)) * (p - x)

    def _naca5_slope(self, x: float) -> float:
        """NACA 5-digit camber line slope.

        Forward camber line (standard 5-digit, e.g., 230xx):
            x <= m:  z_c = (k1/6)(x^3 - 3*m*x^2 + m^2*(3-m)*x)
            x > m:   z_c = (k1*m^3/6)(1 - x)

        Slope:
            x <= m:  dz/dx = (k1/6)(3*x^2 - 6*m*x + m^2*(3-m))
            x > m:   dz/dx = -k1*m^3/6
        """
        m = self.camber_pos
        k1 = self.k1
        if k1 < 1e-10:
            return 0.0

        if x <= m:
            return (k1 / 6.0) * (3.0 * x * x - 6.0 * m * x +
                                  m * m * (3.0 - m))
        else:
            return -k1 * m ** 3 / 6.0

    @classmethod
    def from_naca_string(cls, name: str) -> AirfoilCamber:
        """Parse NACA designation string.

        Supports:
            NACA 4-digit: NACAMPXX (e.g., NACA4415, NACA2412, NACA0012)
            NACA 5-digit: NACALPSXX (e.g., NACA23015, NACA23012)

        Parameters
        ----------
        name : str
            NACA designation (e.g., 'NACA4415').

        Returns
        -------
        AirfoilCamber
        """
        s = name.upper().replace("NACA", "").replace("-", "").replace(" ", "")

        if len(s) == 4:
            # 4-digit: MPXX
            m = int(s[0]) / 100.0  # max camber
            p = int(s[1]) / 10.0   # position
            return cls(name=name, max_camber=m, camber_pos=p, series="4digit")

        elif len(s) == 5:
            # 5-digit: LPSXX
            # L: design CL = L * 3/20
            # P: max camber position = P/20
            # S: 0=standard, 1=reflex
            L = int(s[0])
            P = int(s[1])
            S = int(s[2])

            cl_design = L * 0.15
            m_pos = P * 0.05  # camber position

            # k1 values from NACA tables (standard camber line)
            k1_table = {
                (2, 1, 0): (0.05, 361.4),
                (2, 2, 0): (0.10, 51.64),
                (2, 3, 0): (0.15, 15.957),
                (2, 4, 0): (0.20, 6.643),
                (2, 5, 0): (0.25, 3.230),
            }
            key = (L, P, S)
            if key in k1_table:
                m_actual, k1 = k1_table[key]
            else:
                # Approximate k1 from design CL
                m_actual = m_pos
                k1 = 6.0 * cl_design / (m_actual ** 2 * (3.0 - m_actual)) \
                     if m_actual > 1e-6 else 0.0

            return cls(name=name, max_camber=cl_design/10.0,
                       camber_pos=m_actual, series="5digit",
                       cl_design=cl_design, k1=k1)

        # Unknown format — symmetric
        return cls(name=name)

    @property
    def alpha_0(self) -> float:
        """Zero-lift angle of attack (radians).

        For 4-digit: alpha_0 ≈ -2 * max_camber (thin airfoil theory).
        For 5-digit: alpha_0 ≈ -CL_design / (2*pi).
        """
        if self.series == "5digit":
            return -self.cl_design / (2.0 * np.pi)
        if self.max_camber > 1e-8:
            return -2.0 * self.max_camber
        return 0.0


@dataclass
class PanelAirfoilConfig:
    """Maps CAERO1 panels to airfoil profiles.

    Attributes
    ----------
    panel_airfoils : dict
        Key: CAERO1 EID, Value: (root_airfoil, tip_airfoil) tuple.
        Tip airfoil may be None for constant-section surfaces.
    """
    panel_airfoils: Dict[int, Tuple[AirfoilCamber, Optional[AirfoilCamber]]] = \
        None

    def __post_init__(self):
        if self.panel_airfoils is None:
            self.panel_airfoils = {}

    @classmethod
    def from_config_dict(cls, cfg: dict, caero_panels: dict) -> PanelAirfoilConfig:
        """Create from YAML config dictionary.

        Expected config format:
            airfoils:
              wing:
                root: NACA23015
                tip: NACA23012
              vtail: NACA0012

        Parameters
        ----------
        cfg : dict
            Config dictionary with 'airfoils' key.
        caero_panels : dict
            BDFModel.caero_panels dict (EID → CAERO1).

        Returns
        -------
        PanelAirfoilConfig
        """
        af_cfg = cfg.get("airfoils", {})
        if not af_cfg:
            return cls()

        result = cls()

        # Get surface name → CAERO1 EID mapping from caero_panels
        # Heuristic: wing panels have large Y-span, vtail panels have
        # nonzero Z-component, etc.
        wing_eids = []
        vtail_eids = []
        vtp_eids = []

        for eid, panel in caero_panels.items():
            p1 = np.asarray(panel.p1, dtype=float)
            p4 = np.asarray(panel.p4, dtype=float)
            span_vec = p4 - p1
            if abs(span_vec[2]) > abs(span_vec[1]) * 0.3:
                # Significant Z component → V-tail or VTP
                if abs(span_vec[1]) > 0.1:
                    vtail_eids.append(eid)
                else:
                    vtp_eids.append(eid)
            else:
                wing_eids.append(eid)

        # Assign airfoils
        if "wing" in af_cfg:
            wing_cfg = af_cfg["wing"]
            if isinstance(wing_cfg, str):
                root_af = AirfoilCamber.from_naca_string(wing_cfg)
                tip_af = None
            else:
                root_af = AirfoilCamber.from_naca_string(wing_cfg.get("root", "NACA0012"))
                tip_str = wing_cfg.get("tip", None)
                tip_af = AirfoilCamber.from_naca_string(tip_str) if tip_str else None
            for eid in wing_eids:
                result.panel_airfoils[eid] = (root_af, tip_af)

        if "vtail" in af_cfg:
            vtail_str = af_cfg["vtail"]
            if isinstance(vtail_str, str):
                vtail_af = AirfoilCamber.from_naca_string(vtail_str)
            else:
                vtail_af = AirfoilCamber.from_naca_string(
                    vtail_str.get("root", "NACA0012"))
            for eid in vtail_eids:
                result.panel_airfoils[eid] = (vtail_af, None)

        if "vtp" in af_cfg:
            vtp_str = af_cfg["vtp"]
            if isinstance(vtp_str, str):
                vtp_af = AirfoilCamber.from_naca_string(vtp_str)
            else:
                vtp_af = AirfoilCamber.from_naca_string(
                    vtp_str.get("root", "NACA0012"))
            for eid in vtp_eids:
                result.panel_airfoils[eid] = (vtp_af, None)

        return result


def compute_camber_normalwash(boxes: List[AeroBox],
                               caero_panels: dict,
                               airfoil_config: PanelAirfoilConfig,
                               ) -> np.ndarray:
    """Compute camber normalwash correction for all DLM/VLM boxes.

    The camber normalwash is dz_c/dx evaluated at each box's 3/4-chord
    control point.  This is added to the VLM RHS to account for airfoil
    camber without modifying the panel geometry.

    Equivalent to NASTRAN's DMI W2GJ matrix.

    Parameters
    ----------
    boxes : list of AeroBox
        DLM/VLM boxes.
    caero_panels : dict
        BDFModel.caero_panels (EID → CAERO1).
    airfoil_config : PanelAirfoilConfig
        Airfoil assignment for each CAERO1 panel.

    Returns
    -------
    w_camber : ndarray (n_boxes,)
        Camber normalwash correction.  Add to RHS before solving.
    """
    n = len(boxes)
    w_camber = np.zeros(n)

    if not airfoil_config.panel_airfoils:
        return w_camber

    # Build mapping: box_id → (CAERO1_eid, spanwise_eta, chordwise_xi)
    # For each CAERO1, boxes are numbered sequentially from caero1.eid
    sorted_eids = sorted(caero_panels.keys())

    # Build box_id → CAERO1 EID lookup
    box_to_caero = {}  # box_id → (caero_eid, span_index, chord_index)
    for eid in sorted_eids:
        panel = caero_panels[eid]
        nspan = max(panel.nspan, 1)
        nchord = max(panel.nchord, 1)
        for j in range(nspan):
            for i in range(nchord):
                bid = eid + j * nchord + i
                box_to_caero[bid] = (eid, j, i, nspan, nchord)

    for idx, box in enumerate(boxes):
        bid = box.box_id
        if bid not in box_to_caero:
            continue

        caero_eid, j_span, i_chord, nspan, nchord = box_to_caero[bid]

        if caero_eid not in airfoil_config.panel_airfoils:
            continue

        root_af, tip_af = airfoil_config.panel_airfoils[caero_eid]

        # Chordwise position of 3/4-chord control point
        # Box goes from xi0 to xi1, control point at 3/4 of box
        xi0 = i_chord / nchord
        xi1 = (i_chord + 1) / nchord
        xi_cp = xi0 + 0.75 * (xi1 - xi0)  # 3/4 of this box

        # Spanwise interpolation for airfoil
        eta = (j_span + 0.5) / nspan  # midspan of this strip

        slope_root = root_af.camber_slope(xi_cp)
        if tip_af is not None:
            slope_tip = tip_af.camber_slope(xi_cp)
            slope = slope_root * (1.0 - eta) + slope_tip * eta
        else:
            slope = slope_root

        w_camber[idx] = slope

    return w_camber
