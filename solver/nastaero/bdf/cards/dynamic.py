"""Dynamic analysis BDF card parsers (SOL 112/146)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
from ..field_parser import nastran_int, nastran_float, nastran_string


@dataclass
class TLOAD1:
    """Time-dependent dynamic load.

    TLOAD1 SID EXCITEID DELAY TYPE TID

    Parameters
    ----------
    sid : int
        Load set identification number.
    exciteid : int
        References DAREA set ID for load application points.
    delay : float
        Time delay before load application.
    load_type : int
        Load type: 0=force, 1=displacement (enforced motion).
    tid : int
        References TABLED1 for time-dependent amplitude.
    """
    sid: int = 0
    exciteid: int = 0
    delay: float = 0.0
    load_type: int = 0
    tid: int = 0

    @classmethod
    def from_fields(cls, fields: List[str]) -> TLOAD1:
        t = cls()
        t.sid = nastran_int(fields[1])
        t.exciteid = nastran_int(fields[2])
        if len(fields) > 3:
            t.delay = nastran_float(fields[3])
        if len(fields) > 4:
            t.load_type = nastran_int(fields[4])
        if len(fields) > 5:
            t.tid = nastran_int(fields[5])
        return t


@dataclass
class DLOAD:
    """Dynamic load combination.

    DLOAD SID S S1 L1 S2 L2 ...

    Combines multiple dynamic load sets:
        F(t) = S * sum_i(S_i * F_i(t))

    Parameters
    ----------
    sid : int
        Load set identification number.
    scale : float
        Overall scale factor S.
    scale_factors : list of float
        Individual scale factors S_i.
    load_ids : list of int
        Load set IDs L_i referenced (e.g. TLOAD1 SIDs).
    """
    sid: int = 0
    scale: float = 1.0
    scale_factors: List[float] = field(default_factory=list)
    load_ids: List[int] = field(default_factory=list)

    @classmethod
    def from_fields(cls, fields: List[str]) -> DLOAD:
        d = cls()
        d.sid = nastran_int(fields[1])
        d.scale = nastran_float(fields[2], 1.0)
        # Pairs: S_i, L_i
        i = 3
        while i + 1 < len(fields):
            sf_str = fields[i].strip()
            lid_str = fields[i + 1].strip()
            if not sf_str and not lid_str:
                i += 2
                continue
            if sf_str and lid_str:
                d.scale_factors.append(nastran_float(sf_str))
                d.load_ids.append(nastran_int(lid_str))
            i += 2
        return d


@dataclass
class TABLED1:
    """Tabular function for time-dependent loading.

    TABLED1 TID XAXIS YAXIS
            x1 y1 x2 y2 x3 y3 x4 y4
            ... ENDT

    Defines a piecewise-linear function y(x) from tabulated (x, y) pairs.
    Linear interpolation is used between data points.

    Parameters
    ----------
    tid : int
        Table identification number.
    x : ndarray
        Independent variable (typically time) values.
    y : ndarray
        Dependent variable (amplitude) values.
    """
    tid: int = 0
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def evaluate(self, t: float) -> float:
        """Linear interpolation at time t."""
        return float(np.interp(t, self.x, self.y))

    @classmethod
    def from_fields(cls, fields: List[str]) -> TABLED1:
        t = cls()
        t.tid = nastran_int(fields[1])
        # Fields 2,3 are XAXIS, YAXIS (usually blank) — skip them
        # Data starts at field 9 (after the header line fields)
        # Parse x-y pairs from remaining fields until ENDT
        vals: List[float] = []
        start = 4 if len(fields) > 4 else 2
        for f in fields[start:]:
            s = f.strip().upper()
            if s == "ENDT" or not s:
                if s == "ENDT":
                    break
                continue
            try:
                vals.append(nastran_float(s))
            except (ValueError, TypeError):
                if s == "ENDT":
                    break
        # vals should be pairs: x1, y1, x2, y2, ...
        n_pairs = len(vals) // 2
        if n_pairs > 0:
            t.x = np.array(vals[0::2][:n_pairs])
            t.y = np.array(vals[1::2][:n_pairs])
        return t


@dataclass
class GUST:
    """Aerodynamic gust definition.

    GUST SID DLOAD WG X0 V

    Parameters
    ----------
    sid : int
        Gust set identification number.
    dload_id : int
        References DLOAD set for gust time history.
    wg : float
        Gust velocity scale factor (velocity units).
    x0 : float
        Gust start x-location along flight path.
    v : float
        Vehicle velocity.
    """
    sid: int = 0
    dload_id: int = 0
    wg: float = 0.0
    x0: float = 0.0
    v: float = 0.0

    @classmethod
    def from_fields(cls, fields: List[str]) -> GUST:
        g = cls()
        g.sid = nastran_int(fields[1])
        if len(fields) > 2:
            g.dload_id = nastran_int(fields[2])
        if len(fields) > 3:
            g.wg = nastran_float(fields[3])
        if len(fields) > 4:
            g.x0 = nastran_float(fields[4])
        if len(fields) > 5:
            g.v = nastran_float(fields[5])
        return g


@dataclass
class DAREA:
    """Dynamic load application point.

    DAREA SID P1 C1 A1 P2 C2 A2

    Parameters
    ----------
    sid : int
        Load set identification number.
    entries : list of (node_id, component, scale)
        Each tuple defines a load application point:
        - node_id: Grid point ID
        - component: DOF component (1-6)
        - scale: Scale factor for the load
    """
    sid: int = 0
    entries: List[Tuple[int, int, float]] = field(default_factory=list)

    @classmethod
    def from_fields(cls, fields: List[str]) -> DAREA:
        d = cls()
        d.sid = nastran_int(fields[1])
        # Parse triplets: P, C, A
        i = 2
        while i + 2 < len(fields):
            p_str = fields[i].strip()
            c_str = fields[i + 1].strip()
            a_str = fields[i + 2].strip()
            if not p_str:
                i += 3
                continue
            nid = nastran_int(fields[i])
            comp = nastran_int(fields[i + 1])
            scale = nastran_float(fields[i + 2])
            d.entries.append((nid, comp, scale))
            i += 3
        return d


@dataclass
class FREQ1:
    """Frequency list for frequency response analysis.

    FREQ1 SID F1 DF NDF

    Generates frequencies: F1, F1+DF, F1+2*DF, ..., F1+NDF*DF

    Parameters
    ----------
    sid : int
        Frequency set identification number.
    f1 : float
        Starting frequency (Hz).
    df : float
        Frequency increment (Hz).
    ndf : int
        Number of frequency increments.
    """
    sid: int = 0
    f1: float = 0.0
    df: float = 0.0
    ndf: int = 0

    @classmethod
    def from_fields(cls, fields: List[str]) -> FREQ1:
        f = cls()
        f.sid = nastran_int(fields[1])
        if len(fields) > 2:
            f.f1 = nastran_float(fields[2])
        if len(fields) > 3:
            f.df = nastran_float(fields[3])
        if len(fields) > 4:
            f.ndf = nastran_int(fields[4])
        return f


@dataclass
class TSTEP:
    """Time step definition for transient analysis output.

    TSTEP SID N1 DT1 NO1

    Parameters
    ----------
    sid : int
        Time step set identification number.
    n_steps : int
        Number of time steps.
    dt : float
        Time step size.
    skip : int
        Output interval (every N-th step).
    """
    sid: int = 0
    n_steps: int = 0
    dt: float = 0.0
    skip: int = 1

    @classmethod
    def from_fields(cls, fields: List[str]) -> TSTEP:
        t = cls()
        t.sid = nastran_int(fields[1])
        if len(fields) > 2:
            t.n_steps = nastran_int(fields[2])
        if len(fields) > 3:
            t.dt = nastran_float(fields[3])
        if len(fields) > 4:
            t.skip = nastran_int(fields[4], 1)
        return t


@dataclass
class TABDMP1:
    """Modal damping table.

    TABDMP1 TID TYPE
            f1 g1 f2 g2 ... ENDT

    Parameters
    ----------
    tid : int
        Table identification number.
    damp_type : str
        Damping type: "G" (structural), "CRIT" (critical fraction),
        "Q" (quality factor).
    freqs : ndarray
        Frequency values (Hz).
    values : ndarray
        Damping values at each frequency.
    """
    tid: int = 0
    damp_type: str = "G"
    freqs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    values: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @classmethod
    def from_fields(cls, fields: List[str]) -> TABDMP1:
        t = cls()
        t.tid = nastran_int(fields[1])
        if len(fields) > 2:
            dtype = fields[2].strip().upper()
            if dtype:
                t.damp_type = dtype
        # Data pairs: f1, g1, f2, g2, ... until ENDT
        vals: List[float] = []
        start = 3
        # For fixed-format cards, data may start at field 9 (second line)
        for f in fields[start:]:
            s = f.strip().upper()
            if s == "ENDT":
                break
            if not s:
                continue
            try:
                vals.append(nastran_float(s))
            except (ValueError, TypeError):
                if s == "ENDT":
                    break
        n_pairs = len(vals) // 2
        if n_pairs > 0:
            t.freqs = np.array(vals[0::2][:n_pairs])
            t.values = np.array(vals[1::2][:n_pairs])
        return t

    def get_damping(self, freq: float) -> float:
        """Interpolate damping value at a given frequency."""
        if len(self.freqs) == 0:
            return 0.0
        return float(np.interp(freq, self.freqs, self.values))
