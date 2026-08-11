"""Real-time flight simulator with live structural loads visualization.

Combines 6-DOF rigid-body dynamics (200 Hz) with modal reduced-order
structural response (30 Hz) and PyQt5/PyVista/pyqtgraph visualization (20 Hz).

Architecture:
    SimulationThread (QThread): 200 Hz RK4 dynamics + 30 Hz ModalROM
    Main thread (Qt event loop): 20 Hz GUI updates via QTimer
    SimulatorState: thread-safe shared state with threading.Lock

Usage:
    window = RealtimeSimulatorWindow(bdf_model, rom, params, vn_data)
    window.show()
    QApplication.exec_()
"""
from __future__ import annotations

import math
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSlider, QLabel, QGroupBox, QPushButton, QStatusBar,
    QFrame, QComboBox,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QKeyEvent, QFont

import pyvista as pv
from pyvistaqt import QtInteractor

import pyqtgraph as pg

from ..loads_analysis.certification.flight_sim import (
    AircraftState, AircraftParams, ControlInput,
    six_dof_derivatives, trim_initial_state,
)
from ..loads_analysis.certification.modal_rom import ModalROM
from ..loads_analysis.case_generator import isa_atmosphere
from .mesh_builder import build_structural_mesh, build_beam_tubes


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_G = 9.80665  # m/s²
_DEG = math.degrees(1.0)  # 1 rad in degrees


# ---------------------------------------------------------------------------
# Thread-safe shared state
# ---------------------------------------------------------------------------

@dataclass
class SimulatorState:
    """Thread-safe shared state between simulation and GUI threads.

    Uses a simple lock with latest-value-overwrites semantics
    to avoid queue backlog and latency buildup.
    """
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Control inputs (GUI → sim thread)
    delta_e: float = 0.0    # Elevator (rad)
    delta_a: float = 0.0    # Aileron (rad)
    delta_r: float = 0.0    # Rudder (rad)

    # Rigid-body state (sim → GUI)
    u: float = 0.0
    v: float = 0.0
    w: float = 0.0
    p: float = 0.0
    q: float = 0.0
    r: float = 0.0
    phi: float = 0.0
    theta: float = 0.0
    psi: float = 0.0
    xe: float = 0.0
    ye: float = 0.0
    ze: float = 0.0

    # Derived quantities
    V_tas: float = 0.0
    alpha_deg: float = 0.0
    beta_deg: float = 0.0
    nz: float = 1.0
    altitude_m: float = 0.0
    rho: float = 1.225

    # Structural response (sim → GUI) — arrays for fast mesh updates
    disp_array: Optional[np.ndarray] = None      # (n_nodes, 6)
    force_array: Optional[np.ndarray] = None      # (n_nodes, 6)
    stress_vm: Optional[np.ndarray] = None        # (n_nodes,) von Mises MPa
    max_disp_mm: float = 0.0

    # Timing
    sim_time: float = 0.0
    sim_fps: float = 0.0
    rom_fps: float = 0.0

    # Control flags
    running: bool = True
    paused: bool = False
    reset_to_trim: bool = False  # Signal sim thread to reset state

    def set_controls(self, de: float, da: float, dr: float) -> None:
        with self._lock:
            self.delta_e = de
            self.delta_a = da
            self.delta_r = dr

    def get_controls(self) -> Tuple[float, float, float]:
        with self._lock:
            return self.delta_e, self.delta_a, self.delta_r

    def update_rigid(self, state_arr: np.ndarray, V: float,
                     alpha: float, beta: float, nz: float,
                     sim_time: float, sim_fps: float,
                     altitude: float, rho: float) -> None:
        with self._lock:
            self.u = state_arr[0]
            self.v = state_arr[1]
            self.w = state_arr[2]
            self.p = state_arr[3]
            self.q = state_arr[4]
            self.r = state_arr[5]
            self.phi = state_arr[6]
            self.theta = state_arr[7]
            self.psi = state_arr[8]
            self.xe = state_arr[9]
            self.ye = state_arr[10]
            self.ze = state_arr[11]
            self.V_tas = V
            self.alpha_deg = alpha
            self.beta_deg = beta
            self.nz = nz
            self.sim_time = sim_time
            self.sim_fps = sim_fps
            self.altitude_m = altitude
            self.rho = rho

    def update_structural(self, disp_arr: np.ndarray,
                          force_arr: np.ndarray,
                          max_d: float, rom_fps: float,
                          stress_vm: Optional[np.ndarray] = None) -> None:
        with self._lock:
            self.disp_array = disp_arr
            self.force_array = force_arr
            self.stress_vm = stress_vm
            self.max_disp_mm = max_d
            self.rom_fps = rom_fps

    def get_snapshot(self) -> dict:
        """Get a snapshot of all state for GUI rendering."""
        with self._lock:
            return {
                'V_tas': self.V_tas,
                'alpha_deg': self.alpha_deg,
                'beta_deg': self.beta_deg,
                'nz': self.nz,
                'p': self.p, 'q': self.q, 'r': self.r,
                'phi': self.phi, 'theta': self.theta, 'psi': self.psi,
                'altitude_m': self.altitude_m,
                'rho': self.rho,
                'sim_time': self.sim_time,
                'sim_fps': self.sim_fps,
                'rom_fps': self.rom_fps,
                'disp_array': self.disp_array,
                'force_array': self.force_array,
                'stress_vm': self.stress_vm,
                'max_disp_mm': self.max_disp_mm,
                'delta_e': self.delta_e,
                'delta_a': self.delta_a,
                'delta_r': self.delta_r,
                'paused': self.paused,
            }


# ---------------------------------------------------------------------------
# Simulation thread
# ---------------------------------------------------------------------------

class SimulationThread(QThread):
    """Background thread running 6-DOF dynamics + ModalROM.

    Inner loop: RK4 @ 200 Hz for rigid-body dynamics.
    Mid loop: ModalROM @ 30 Hz for structural response.

    Parameters
    ----------
    params : AircraftParams
        Aircraft 6-DOF parameters.
    initial_state : AircraftState
        Trim initial condition.
    delta_e_trim : float
        Trim elevator deflection (rad).
    rom : ModalROM
        Pre-built modal reduced-order model.
    shared : SimulatorState
        Thread-safe shared state.
    altitude_m : float
        Simulation altitude.
    """

    def __init__(self, params: AircraftParams,
                 initial_state: AircraftState,
                 delta_e_trim: float,
                 rom: ModalROM,
                 shared: SimulatorState,
                 altitude_m: float = 0.0,
                 elem_rows: Optional[np.ndarray] = None):
        super().__init__()
        self.params = params
        self.initial_y = initial_state.to_array().copy()
        self.delta_e_trim = delta_e_trim
        self.rom = rom
        self.shared = shared
        self.altitude_m = altitude_m
        self.elem_rows = elem_rows  # element-only row mask for max_d

        rho, _, _ = isa_atmosphere(altitude_m)
        self.rho = rho
        self.params.rho = rho

        self.dt = 0.005  # 200 Hz
        self.rom_every = 7  # ROM every 7 steps ≈ 30 Hz

    def run(self):
        y = self.initial_y.copy()
        t = 0.0
        step = 0
        fps_counter = 0
        fps_time = time.perf_counter()
        rom_counter = 0
        rom_time = time.perf_counter()

        while self.shared.running:
            if self.shared.paused:
                time.sleep(0.01)
                continue

            # Check for reset-to-trim request
            if self.shared.reset_to_trim:
                y = self.initial_y.copy()
                t = 0.0
                step = 0
                self.shared.reset_to_trim = False

            loop_start = time.perf_counter()

            # Get current control inputs
            de, da, dr = self.shared.get_controls()
            # Add trim elevator offset
            de_total = de + self.delta_e_trim

            ctrl = ControlInput(delta_e=de_total, delta_a=da, delta_r=dr)

            # RK4 step
            h = self.dt
            k1 = six_dof_derivatives(y, t, self.params, ctrl)
            k2 = six_dof_derivatives(y + 0.5 * h * k1, t + 0.5 * h, self.params, ctrl)
            k3 = six_dof_derivatives(y + 0.5 * h * k2, t + 0.5 * h, self.params, ctrl)
            k4 = six_dof_derivatives(y + h * k3, t + h, self.params, ctrl)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
            step += 1

            # Compute derived quantities
            u, v, w = y[0], y[1], y[2]
            V = math.sqrt(u**2 + v**2 + w**2)
            alpha = math.atan2(w, max(u, 1.0))
            beta = math.asin(max(-1.0, min(1.0, v / max(V, 1.0))))

            # Compute nz from aero forces
            d = self.params.derivs
            qbar = 0.5 * self.params.rho * V**2
            _ALPHA_STALL = 0.30
            alpha_aero = max(-_ALPHA_STALL, min(_ALPHA_STALL, alpha))
            q_hat = y[4] * self.params.c_bar / (2.0 * max(V, 1.0))
            CL = d.CLalpha * alpha_aero + d.CLdelta_e * de_total + d.CLq * q_hat
            nz = qbar * self.params.S * CL / (self.params.mass * _G)

            # FPS calculation
            fps_counter += 1
            now = time.perf_counter()
            if now - fps_time >= 0.5:
                sim_fps = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now
            else:
                sim_fps = self.shared.sim_fps

            # Update shared rigid-body state
            self.shared.update_rigid(
                y, V, math.degrees(alpha), math.degrees(beta),
                nz, t, sim_fps, self.altitude_m, self.rho,
            )

            # ModalROM response (every rom_every steps ≈ 30 Hz)
            if step % self.rom_every == 0 and self.rom is not None:
                rom_start = time.perf_counter()
                try:
                    disp_arr, force_arr = self.rom.compute_response_arrays(
                        alpha=alpha,
                        V=V,
                        de=de_total,
                        da=da,
                        dr=dr,
                        nz=nz,
                        rho=self.rho,
                        beta=beta,
                        viz=False,  # raw FEA displacement (no modal filter)
                        # Modal filter produced local modes at CONM2/spline
                        # nodes. Spanwise bending interpolation in GUI
                        # handles smoothing instead.
                    )
                    # Compute von Mises stress (< 0.3 ms)
                    stress_vm = self.rom.compute_stress_von_mises(
                        alpha=alpha, V=V, de=de_total, da=da,
                        dr=dr, nz=nz, rho=self.rho,
                    )
                    # Compute max displacement at element nodes only
                    # (excludes CONM2 mass-only nodes that inflate max)
                    all_mag = np.linalg.norm(disp_arr[:, :3], axis=1)
                    if (self.elem_rows is not None
                            and len(self.elem_rows) > 0):
                        max_d = float(np.max(all_mag[self.elem_rows]))
                    else:
                        max_d = float(np.max(all_mag))

                    rom_counter += 1
                    rom_elapsed = time.perf_counter() - rom_start
                    r_fps = 1.0 / max(rom_elapsed, 1e-6)

                    self.shared.update_structural(
                        disp_arr, force_arr, max_d, r_fps,
                        stress_vm=stress_vm)
                except Exception as e:
                    pass  # ROM failure — keep previous state

            # Real-time pacing: sleep to maintain 200 Hz
            elapsed = time.perf_counter() - loop_start
            sleep_time = self.dt - elapsed
            if sleep_time > 0.0005:
                time.sleep(sleep_time)


# ---------------------------------------------------------------------------
# Main GUI window
# ---------------------------------------------------------------------------

class RealtimeSimulatorWindow(QMainWindow):
    """Real-time flight simulator window with live structural loads.

    Layout:
        Left:  3D viewport (pyvistaqt.QtInteractor)
        Right: Control sliders + time history + V-n diagram + status

    Parameters
    ----------
    bdf_model : BDFModel
        Parsed BDF model for mesh building.
    rom : ModalROM
        Pre-built modal ROM.
    params : AircraftParams
        Aircraft parameters.
    vn_data : dict or None
        V-n diagram data for overlay plot.
    altitude_m : float
        Simulation altitude.
    """

    def __init__(self, bdf_model, rom: ModalROM,
                 params: AircraftParams,
                 vn_data=None,
                 altitude_m: float = 0.0,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("NastAero Real-Time Flight Simulator")
        self.setFixedSize(1600, 900)

        self.bdf_model = bdf_model
        self.rom = rom
        self.params = params
        self.vn_data = vn_data
        self.altitude_m = altitude_m

        # Shared state
        self.shared = SimulatorState()

        # Trim initial condition
        rho, _, _ = isa_atmosphere(altitude_m)
        params.rho = rho
        V_cruise = 80.0  # m/s default cruise speed
        initial_state, delta_e_trim = trim_initial_state(params, V_cruise, nz=1.0)
        self.initial_state = initial_state
        self.delta_e_trim = delta_e_trim

        # Set initial controls to zero (trim elevator handled separately)
        self.shared.set_controls(0.0, 0.0, 0.0)

        # Time history buffers (rolling 10s window at 20 Hz = 200 points)
        self._history_len = 200
        self._t_hist = np.zeros(self._history_len)
        self._nz_hist = np.ones(self._history_len)
        self._alpha_hist = np.zeros(self._history_len)
        self._beta_hist = np.zeros(self._history_len)
        self._p_hist = np.zeros(self._history_len)
        self._q_hist = np.zeros(self._history_len)
        self._r_hist = np.zeros(self._history_len)
        self._vn_trail_V = []
        self._vn_trail_n = []

        # Build mesh
        self._sorted_nids = sorted(bdf_model.nodes.keys())
        self._nid_to_idx = {nid: i for i, nid in enumerate(self._sorted_nids)}
        self._base_points = np.zeros((len(self._sorted_nids), 3))
        for i, nid in enumerate(self._sorted_nids):
            self._base_points[i] = bdf_model.nodes[nid].xyz_global

        # -----------------------------------------------------------
        # Classify nodes: element-connected vs mass-only, wing vs body
        # -----------------------------------------------------------
        _elem_nid_set = set()
        for eid, elem in bdf_model.elements.items():
            if hasattr(elem, 'node_ids'):
                for n in elem.node_ids:
                    _elem_nid_set.add(n)
        n_total = len(self._sorted_nids)
        # Boolean mask: True for nodes that belong to at least one element
        self._elem_mask = np.array(
            [nid in _elem_nid_set for nid in self._sorted_nids], dtype=bool)
        # Wing node mask: element nodes with |Y| > 2000 mm
        self._wing_mask = np.array(
            [(nid in _elem_nid_set and abs(self._base_points[i, 1]) > 2000)
             for i, nid in enumerate(self._sorted_nids)], dtype=bool)
        print(f"       Node classification: {np.sum(self._elem_mask)} element, "
              f"{n_total - np.sum(self._elem_mask)} mass-only, "
              f"{np.sum(self._wing_mask)} wing")

        # Pre-built mapping: rom node IDs → mesh point indices for vectorized updates
        rom_nids = rom.sorted_nids  # sorted node ID list from ModalROM
        self._rom_mesh_idx = np.array(
            [self._nid_to_idx[nid] for nid in rom_nids
             if nid in self._nid_to_idx], dtype=np.intp)
        self._rom_nid_list = np.array(
            [nid for nid in rom_nids if nid in self._nid_to_idx], dtype=np.int64)

        # Which ROM output rows map to wing nodes (for auto-scale and colormap)
        n_map = len(self._rom_mesh_idx)
        self._rom_wing_rows = np.array(
            [r for r in range(n_map)
             if self._wing_mask[self._rom_mesh_idx[r]]], dtype=np.intp)
        self._rom_elem_rows = np.array(
            [r for r in range(n_map)
             if self._elem_mask[self._rom_mesh_idx[r]]], dtype=np.intp)

        # Deformation scale — auto-compute from WING TIP displacement
        # (not global max which is dominated by fuselage mass items)
        semi_span = float(np.max(np.abs(self._base_points[:, 1])))
        self._semi_span = semi_span
        try:
            _probe_arr, _ = rom.compute_response_arrays(
                alpha=0.05, V=80.0, de=-0.04, nz=2.5, rho=1.225,
                viz=False)  # raw FEA displacement (no modal filter)
            # Max displacement at wing nodes only
            if len(self._rom_wing_rows) > 0:
                _wing_disp = np.linalg.norm(
                    _probe_arr[self._rom_wing_rows, :3], axis=1)
                _max_wing = float(np.max(_wing_disp))
            else:
                _max_wing = 0.0
            _max_all = float(np.max(np.linalg.norm(
                _probe_arr[:, :3], axis=1)))
            print(f"       Probe (2.5g): max_all={_max_all:.4f} mm, "
                  f"max_wing={_max_wing:.4f} mm")
            # Target: wing tip visible deflection = 5% of semi-span
            if _max_wing > 1e-12:
                self._disp_scale = min(
                    max(0.05 * semi_span / _max_wing, 10.0), 50000.0)
            elif _max_all > 1e-12:
                self._disp_scale = min(
                    max(0.02 * semi_span * 2 / _max_all, 10.0), 50000.0)
            else:
                self._disp_scale = 5000.0
        except Exception:
            self._disp_scale = 5000.0
        self._disp_scale = round(self._disp_scale)

        # -----------------------------------------------------------
        # Deformation blend: node-ID-based component classification.
        # GACOMP node ID ranges (from BDF model):
        #   100k-299k = fuselage → 0 (suppress completely)
        #   300k-499k = wings    → 1.0 (full, with root fade)
        #   500k-699k = HTP      → 1.0 (full)
        #   700k-799k = VTP      → 1.0 (full)
        # -----------------------------------------------------------
        n_total = len(self._sorted_nids)
        self._deform_blend = np.zeros(n_total)
        n_fuse = n_wing = n_htp = n_vtp = n_other = 0
        for i, nid in enumerate(self._sorted_nids):
            if 100000 <= nid < 300000:
                # Fuselage: zero deformation — prevents mass-item spikes
                self._deform_blend[i] = 0.0
                n_fuse += 1
            elif 300000 <= nid < 500000:
                # Wing: full deformation with smooth root fade-in
                y = abs(self._base_points[i, 1])
                if y > 1500:
                    self._deform_blend[i] = 1.0
                elif y > 500:
                    self._deform_blend[i] = (y - 500) / 1000.0
                else:
                    self._deform_blend[i] = 0.0  # wing root at fuselage junction
                n_wing += 1
            elif 500000 <= nid < 700000:
                # HTP: full deformation
                self._deform_blend[i] = 1.0
                n_htp += 1
            elif 700000 <= nid < 800000:
                # VTP: full deformation
                self._deform_blend[i] = 1.0
                n_vtp += 1
            else:
                self._deform_blend[i] = 0.0
                n_other += 1
        print(f"       Deformation blend by component: "
              f"fuse={n_fuse}(off), wing={n_wing}, htp={n_htp}, "
              f"vtp={n_vtp}, other={n_other}")

        # -----------------------------------------------------------
        # Spanwise bending interpolation for visible wing/tail bending.
        # Wing skin shell nodes have tiny FEA displacement (locally stiff).
        # Global bending is captured by spar CBAR/CBEAM nodes.
        # Interpolate spar z-displacement → all skin nodes at each span
        # station for smooth, visible wing bending visualization.
        # -----------------------------------------------------------
        _bar_nids = set()
        for eid, elem in bdf_model.elements.items():
            etype = type(elem).__name__
            if etype in ('CBAR', 'CBEAM'):
                for nid in elem.node_ids:
                    _bar_nids.add(nid)

        rom_nids = rom.sorted_nids

        def _build_bending_data(nid_lo, nid_hi, span_axis):
            """Build spar and skin node arrays for one lifting surface half.

            Parameters
            ----------
            nid_lo, nid_hi : int
                Node ID range [lo, hi).
            span_axis : int
                0=X, 1=Y, 2=Z — which axis is the span direction.
            """
            spar_list = []   # (|span_coord|, rom_row_index)
            skin_rows = []
            skin_span = []
            for r, nid in enumerate(rom_nids):
                if nid_lo <= nid < nid_hi and nid in self._nid_to_idx:
                    mesh_i = self._nid_to_idx[nid]
                    span_val = abs(self._base_points[mesh_i, span_axis])
                    if nid in _bar_nids:
                        spar_list.append((span_val, r))
                    skin_rows.append(r)
                    skin_span.append(span_val)
            if len(spar_list) < 3:
                return None
            spar_list.sort(key=lambda x: x[0])
            return {
                'spar_span': np.array([s[0] for s in spar_list]),
                'spar_rows': np.array([s[1] for s in spar_list], dtype=np.intp),
                'skin_rows': np.array(skin_rows, dtype=np.intp),
                'skin_span': np.array(skin_span),
            }

        self._bending_surfaces = []
        # Left wing (300k-399k), span along Y
        bd = _build_bending_data(300000, 400000, 1)
        if bd:
            self._bending_surfaces.append(('L Wing', bd))
        # Right wing (400k-499k), span along Y
        bd = _build_bending_data(400000, 500000, 1)
        if bd:
            self._bending_surfaces.append(('R Wing', bd))
        # Left HTP (500k-599k), span along Y
        bd = _build_bending_data(500000, 600000, 1)
        if bd:
            self._bending_surfaces.append(('L HTP', bd))
        # Right HTP (600k-699k), span along Y
        bd = _build_bending_data(600000, 700000, 1)
        if bd:
            self._bending_surfaces.append(('R HTP', bd))
        # VTP (700k-799k), span along Z
        bd = _build_bending_data(700000, 800000, 2)
        if bd:
            self._bending_surfaces.append(('VTP', bd))

        for name, bd in self._bending_surfaces:
            print(f"       Bending interp: {name} — "
                  f"{len(bd['spar_rows'])} spar, "
                  f"{len(bd['skin_rows'])} total nodes")

        # -----------------------------------------------------------
        # Recompute auto-scale with bending interpolation applied
        # -----------------------------------------------------------
        try:
            _probe_arr, _ = rom.compute_response_arrays(
                alpha=0.05, V=80.0, de=-0.04, nz=2.5, rho=1.225,
                viz=False)  # raw FEA displacement
            _probe_disp = _probe_arr[:len(self._rom_mesh_idx), :3].copy()
            # Apply bending interpolation to probe
            for _name, bd in self._bending_surfaces:
                spar_dz = _probe_arr[bd['spar_rows'], 2]
                interp_dz = np.interp(
                    bd['skin_span'], bd['spar_span'], spar_dz)
                _probe_disp[bd['skin_rows'], 2] = interp_dz
            # Max at wing nodes (with interpolated bending)
            if len(self._rom_wing_rows) > 0:
                _wing_disp = np.linalg.norm(
                    _probe_disp[self._rom_wing_rows], axis=1)
                _max_wing = float(np.max(_wing_disp))
            else:
                _max_wing = 0.0
            _max_all = float(np.max(np.linalg.norm(
                _probe_disp, axis=1)))
            print(f"       Probe (2.5g) with bending: max_all={_max_all:.4f} mm, "
                  f"max_wing={_max_wing:.4f} mm")
            # Target: wing tip visible deflection = 5% of semi-span
            if _max_wing > 1e-12:
                self._disp_scale = min(
                    max(0.05 * semi_span / _max_wing, 10.0), 50000.0)
            elif _max_all > 1e-12:
                self._disp_scale = min(
                    max(0.02 * semi_span * 2 / _max_all, 10.0), 50000.0)
            else:
                self._disp_scale = 5000.0
            self._disp_scale = round(self._disp_scale)
            print(f"       Auto-scale (bending): {self._disp_scale}×")
        except Exception as e:
            print(f"       Auto-scale recompute failed: {e}")

        # -----------------------------------------------------------
        # Synthetic wing bending: nz-driven physical bending shape.
        # Uses GEOMETRY (not node IDs) to classify components, so
        # works with any BDF model (GACOMP, UAM, etc.).
        # Target: 3% of semi-span deflection at wing tip per unit Δnz.
        # -----------------------------------------------------------
        target_tip_mm = 0.03 * semi_span   # ≈168 mm per unit Δnz

        # Detect wing root/tip from actual Y coordinates
        pts = self._base_points
        y_all = np.abs(pts[:, 1])
        x_all = pts[:, 0]
        z_all = pts[:, 2]
        y_span = float(np.max(y_all))
        z_span = float(np.max(z_all) - np.min(z_all))

        # Centerline tolerance: 6% of Y-span (same as component_id.py)
        cl_tol = max(0.06 * y_span, 50.0)

        # Classify by geometry:
        # Lateral nodes (|Y| > cl_tol) = wing or tail
        # High-Z lateral = tail, normal = wing
        # Centerline = fuselage or VTP (high Z)
        is_lateral = y_all > cl_tol
        is_centerline = ~is_lateral

        # Among lateral nodes, separate wing vs tail by X gap
        lat_x = x_all[is_lateral]
        if len(lat_x) > 20:
            x_median = float(np.median(lat_x))
            x_range = float(np.max(lat_x) - np.min(lat_x))
            x_gap = x_median + 0.3 * x_range
            is_wing = is_lateral & (x_all < x_gap)
            is_tail_lat = is_lateral & (x_all >= x_gap)
        else:
            is_wing = is_lateral
            is_tail_lat = np.zeros(len(pts), dtype=bool)

        # L/R split
        is_lwing = is_wing & (pts[:, 1] < 0)
        is_rwing = is_wing & (pts[:, 1] >= 0)

        # VTP: centerline nodes at high Z
        z_center = z_all[is_centerline]
        z_high_thresh = float(np.median(z_center) + 0.5 * (np.max(z_center) - np.min(z_center))) if len(z_center) > 5 else 9999
        is_vtp = is_centerline & (z_all > z_high_thresh)

        # Wing root/tip Y from actual geometry
        wing_y = y_all[is_wing]
        if len(wing_y) > 5:
            y_root = float(np.percentile(wing_y, 5))
            y_tip = float(np.percentile(wing_y, 98))
        else:
            y_root = cl_tol
            y_tip = semi_span

        # Tail root/tip Y
        tail_y = y_all[is_tail_lat]
        if len(tail_y) > 5:
            htp_semi_span = float(np.percentile(tail_y, 98))
        else:
            htp_semi_span = y_tip * 0.3

        # VTP root/tip Z
        vtp_z = z_all[is_vtp]
        if len(vtp_z) > 3:
            vtp_root_z = float(np.min(vtp_z))
            vtp_tip_z = float(np.max(vtp_z))
        else:
            vtp_root_z = z_high_thresh
            vtp_tip_z = vtp_root_z + 1000

        n_total = len(self._sorted_nids)
        self._synth_bend_z = np.zeros(n_total)  # dz per unit Δnz
        self._synth_bend_y = np.zeros(n_total)  # dy per unit Δβ (VTP)
        n_synth_wing = n_synth_htp = n_synth_vtp = 0

        for i in range(n_total):
            if is_wing[i]:
                # Wing — cubic cantilever: η²(3−η)/2
                y = y_all[i]
                if y > y_root:
                    eta = min((y - y_root) / max(y_tip - y_root, 1.0), 1.0)
                    self._synth_bend_z[i] = target_tip_mm * eta**2 * (3.0 - eta) / 2.0
                    n_synth_wing += 1
            elif is_tail_lat[i]:
                # HTP — 40% amplitude of wing
                y = y_all[i]
                eta = min(y / max(htp_semi_span, 1.0), 1.0)
                self._synth_bend_z[i] = target_tip_mm * 0.4 * eta**2 * (3.0 - eta) / 2.0
                n_synth_htp += 1
            elif is_vtp[i]:
                # VTP — bending in Y from sideslip
                z = z_all[i]
                if z > vtp_root_z:
                    eta = min((z - vtp_root_z) / max(vtp_tip_z - vtp_root_z, 1.0), 1.0)
                    self._synth_bend_y[i] = target_tip_mm * 0.3 * eta**2 * (3.0 - eta) / 2.0
                    n_synth_vtp += 1

        # Boolean masks for aileron differential bending (geometry-based)
        self._lwing_mask = is_lwing
        self._rwing_mask = is_rwing

        print(f"       Synthetic bending: wing={n_synth_wing}, htp={n_synth_htp}, "
              f"vtp={n_synth_vtp} nodes")
        print(f"       Wing Y range: root={y_root:.0f} tip={y_tip:.0f} mm")
        print(f"       Target tip deflection: {target_tip_mm:.0f} mm/Δnz "
              f"(= {target_tip_mm/semi_span*100:.1f}% semi-span)")

        # -----------------------------------------------------------
        # Control surface deflection: identify TE nodes aft of rear
        # spar (hinge line) and pre-compute rotation arms.
        # Nodes aft of rear_spar_frac * local_chord rotate about the
        # hinge line.  dz = (x - x_hinge) * sin(delta).
        # -----------------------------------------------------------
        wing_spar_frac = 0.65   # rear spar at 65% chord
        tail_spar_frac = 0.60   # rear spar at 60% chord

        # Per-node: rotation arm (x - x_hinge), zero for non-CS nodes
        self._cs_arm_elev = np.zeros(n_total)   # elevator (tail TE)
        self._cs_arm_ail_r = np.zeros(n_total)  # right aileron (R wing TE)
        self._cs_arm_ail_l = np.zeros(n_total)  # left aileron (L wing TE)
        n_elev = n_ail = 0

        # Classify by Y-station: for each station find local chord,
        # then mark nodes aft of hinge
        for comp_mask, spar_frac, arm_arr, is_ail, y_outboard_frac in [
            (is_tail_lat, tail_spar_frac, '_cs_arm_elev', False, 0.0),
            (is_rwing, wing_spar_frac, '_cs_arm_ail_r', True, 0.5),
            (is_lwing, wing_spar_frac, '_cs_arm_ail_l', True, 0.5),
        ]:
            idxs = np.where(comp_mask)[0]
            if len(idxs) < 4:
                continue
            comp_x = pts[idxs, 0]
            comp_y = pts[idxs, 1]
            # Group by Y-station (round to 10mm)
            y_round = np.round(comp_y, -1)
            for y_st in sorted(set(y_round)):
                st_mask = y_round == y_st
                st_idxs = idxs[st_mask]
                st_x = pts[st_idxs, 0]
                if len(st_x) < 2:
                    continue
                local_chord = st_x.max() - st_x.min()
                if local_chord < 10:
                    continue
                x_hinge = st_x.min() + spar_frac * local_chord

                # Aileron: only outboard half of wing
                if is_ail and y_outboard_frac > 0:
                    y_half = np.percentile(np.abs(comp_y), 50)
                    if abs(pts[st_idxs[0], 1]) < y_half:
                        continue

                for idx in st_idxs:
                    x_node = pts[idx, 0]
                    if x_node > x_hinge + 5:  # 5mm tolerance
                        arm = x_node - x_hinge
                        arr = getattr(self, arm_arr)
                        arr[idx] = arm
                        if is_ail:
                            n_ail += 1
                        else:
                            n_elev += 1

        print(f"       Control surfaces: elev={n_elev}, ail={n_ail} TE nodes")

        # -----------------------------------------------------------
        # Force visualization: pre-compute mapping from ROM force_arr
        # rows to mesh point indices for fast scalar updates.
        # -----------------------------------------------------------
        self._display_mode = 0  # 0=displacement, 1=Fz, 2=|F|
        self._arrow_scale_pct = 0  # 0=off, 1-100 = aero arrow scale

        # Force array → mesh point mapping (same as displacement mapping)
        # force_arr is (n_rom_nodes, 6) ordered by rom.sorted_nids
        # We need to map each ROM row → mesh point index
        self._force_scalars = np.zeros(n_total)  # per-mesh-point force value

        # Pre-compute bbox diagonal for arrow auto-scale
        bbox = self._base_points.max(axis=0) - self._base_points.min(axis=0)
        self._bbox_diag = float(np.linalg.norm(bbox))

        # Pre-compute force arrow infrastructure
        self._arrow_max = 50  # max structural force arrows

        # Pre-compute aero panel centers and data for panel force arrows
        self._aero_centers = np.array(
            [b.control_point for b in rom.boxes])  # (n_boxes, 3)
        self._aero_normals = rom.box_normals.copy()  # (n_boxes, 3)
        self._aero_areas = rom.box_areas.copy()
        self._aero_chords = rom.box_chords.copy()
        self._aero_D_inv = rom.D_inv   # AIC inverse
        self._aero_w_alpha = rom.w_alpha.copy()
        self._aero_w_elev = rom.w_elev.copy()
        self._aero_w_ail = rom.w_ail.copy()
        self._aero_w_rud = rom.w_rud.copy()
        self._aero_arrow_actor = None

        # Pre-compute inertial (mass) node data for weight arrows
        self._inertia_arrow_actor = None
        self._inertia_arrow_scale_pct = 0  # separate slider
        mass_nids = []
        mass_positions = []
        mass_values = []
        for nid, mass in rom.node_masses.items():
            if mass > 1e-10 and nid in self._nid_to_idx:
                mass_nids.append(nid)
                mass_positions.append(self._base_points[self._nid_to_idx[nid]])
                mass_values.append(mass)
        self._mass_node_positions = np.array(mass_positions) if mass_positions else np.zeros((0, 3))
        self._mass_node_values = np.array(mass_values) if mass_values else np.zeros(0)
        self._mass_g_accel = rom.g_accel  # mm/s² (9810 for N-mm-sec)
        n_mass = len(mass_nids)

        print(f"       Force visualization ready "
              f"(bbox_diag={self._bbox_diag:.0f} mm, "
              f"{len(rom.boxes)} aero panels, "
              f"{n_mass} mass nodes, total mass={np.sum(self._mass_node_values):.1f} kg)")

        self._setup_ui()
        self._setup_3d_viewport()
        self._setup_plots()
        self._setup_vn_diagram()

        # FPS counter for GUI
        self._gui_frame_count = 0
        self._gui_fps_time = time.perf_counter()
        self._gui_fps = 0.0

        # Start simulation thread
        self._sim_thread = SimulationThread(
            params, initial_state, delta_e_trim, rom, self.shared, altitude_m,
            elem_rows=self._rom_elem_rows)
        self._sim_thread.start()

        # GUI update timer @ 20 Hz
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(50)  # 50 ms = 20 Hz

    def _setup_ui(self):
        """Build the main window layout."""
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout: left (3D, fixed) | right (controls + plots, fixed)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(2)

        # Left: 3D viewport — fixed width
        self._viewport_frame = QFrame()
        self._viewport_frame.setFixedWidth(940)
        self._viewport_layout = QVBoxLayout(self._viewport_frame)
        self._viewport_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self._viewport_frame)

        # Right panel — fixed width
        right_panel = QWidget()
        right_panel.setFixedWidth(650)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        # --- Control sliders ---
        ctrl_group = QGroupBox("Control Surfaces")
        ctrl_layout = QVBoxLayout(ctrl_group)

        # Elevator
        self._elev_label = QLabel("Elevator: 0.0°")
        self._elev_slider = QSlider(Qt.Horizontal)
        self._elev_slider.setRange(-250, 250)  # ±25° × 10
        self._elev_slider.setValue(0)
        self._elev_slider.valueChanged.connect(self._on_elev_changed)
        ctrl_layout.addWidget(self._elev_label)
        ctrl_layout.addWidget(self._elev_slider)

        # Aileron
        self._ail_label = QLabel("Aileron: 0.0°")
        self._ail_slider = QSlider(Qt.Horizontal)
        self._ail_slider.setRange(-200, 200)  # ±20° × 10
        self._ail_slider.setValue(0)
        self._ail_slider.valueChanged.connect(self._on_ail_changed)
        ctrl_layout.addWidget(self._ail_label)
        ctrl_layout.addWidget(self._ail_slider)

        # Rudder
        self._rud_label = QLabel("Rudder: 0.0°")
        self._rud_slider = QSlider(Qt.Horizontal)
        self._rud_slider.setRange(-250, 250)  # ±25° × 10
        self._rud_slider.setValue(0)
        self._rud_slider.valueChanged.connect(self._on_rud_changed)
        ctrl_layout.addWidget(self._rud_label)
        ctrl_layout.addWidget(self._rud_slider)

        # Deformation scale
        init_scale = int(self._disp_scale)
        self._scale_label = QLabel(f"Deformation Scale: {init_scale}×")
        self._scale_slider = QSlider(Qt.Horizontal)
        self._scale_slider.setRange(1, 50000)
        self._scale_slider.setValue(init_scale)
        self._scale_slider.valueChanged.connect(self._on_scale_changed)
        ctrl_layout.addWidget(self._scale_label)
        ctrl_layout.addWidget(self._scale_slider)

        # Display mode: colormap selector
        self._display_label = QLabel("Display:")
        self._display_combo = QComboBox()
        self._display_combo.addItems([
            "Displacement (mm)",
            "Force Fz (N)",
            "Force |F| (N)",
            "Stress (Von Mises MPa)",
        ])
        self._display_combo.currentIndexChanged.connect(self._on_display_changed)
        ctrl_layout.addWidget(self._display_label)
        ctrl_layout.addWidget(self._display_combo)

        # Aero force arrows scale (0 = off, 1-100 = scale)
        self._arrow_label = QLabel("Aero Arrows (blue): Off")
        self._arrow_slider = QSlider(Qt.Horizontal)
        self._arrow_slider.setRange(0, 100)
        self._arrow_slider.setValue(0)
        self._arrow_slider.valueChanged.connect(self._on_arrow_scale_changed)
        ctrl_layout.addWidget(self._arrow_label)
        ctrl_layout.addWidget(self._arrow_slider)

        # Inertial force arrows scale (0 = off, 1-100 = scale)
        self._inertia_label = QLabel("Inertial Arrows (red): Off")
        self._inertia_slider = QSlider(Qt.Horizontal)
        self._inertia_slider.setRange(0, 100)
        self._inertia_slider.setValue(0)
        self._inertia_slider.valueChanged.connect(
            self._on_inertia_scale_changed)
        ctrl_layout.addWidget(self._inertia_label)
        ctrl_layout.addWidget(self._inertia_slider)

        # Buttons
        btn_layout = QHBoxLayout()
        self._pause_btn = QPushButton("⏸ Pause")
        self._pause_btn.clicked.connect(self._toggle_pause)
        self._reset_btn = QPushButton("✈ 1g Trim")
        self._reset_btn.setToolTip("Return to 1g level flight trim condition")
        self._reset_btn.clicked.connect(self._return_to_trim)
        btn_layout.addWidget(self._pause_btn)
        btn_layout.addWidget(self._reset_btn)
        ctrl_layout.addLayout(btn_layout)

        right_layout.addWidget(ctrl_group)

        # --- Time history plots (pyqtgraph) ---
        plot_group = QGroupBox("Time History")
        plot_layout = QVBoxLayout(plot_group)

        # nz / alpha / beta plot — fixed height to prevent jitter
        self._plot_nz = pg.PlotWidget(title="Load Factor & Angles")
        self._plot_nz.setLabel('left', 'nz / deg')
        self._plot_nz.setLabel('bottom', 'Time', 's')
        self._plot_nz.addLegend(offset=(50, 5))
        self._plot_nz.setYRange(-2, 5)
        self._plot_nz.setFixedHeight(150)
        self._plot_nz.enableAutoRange(axis='x', enable=False)
        self._curve_nz = self._plot_nz.plot(pen=pg.mkPen('r', width=2), name='nz')
        self._curve_alpha = self._plot_nz.plot(pen=pg.mkPen('b', width=2), name='α°')
        self._curve_beta = self._plot_nz.plot(pen=pg.mkPen('g', width=2), name='β°')
        plot_layout.addWidget(self._plot_nz)

        # Angular rates plot — fixed height
        self._plot_rates = pg.PlotWidget(title="Angular Rates (deg/s)")
        self._plot_rates.setLabel('left', 'deg/s')
        self._plot_rates.setLabel('bottom', 'Time', 's')
        self._plot_rates.addLegend(offset=(50, 5))
        self._plot_rates.setFixedHeight(150)
        self._plot_rates.enableAutoRange(axis='x', enable=False)
        self._curve_p = self._plot_rates.plot(pen=pg.mkPen('r', width=2), name='p')
        self._curve_q = self._plot_rates.plot(pen=pg.mkPen('b', width=2), name='q')
        self._curve_r = self._plot_rates.plot(pen=pg.mkPen('g', width=2), name='r')
        plot_layout.addWidget(self._plot_rates)

        right_layout.addWidget(plot_group)

        # --- V-n diagram placeholder ---
        vn_group = QGroupBox("V-n Diagram")
        self._vn_layout = QVBoxLayout(vn_group)
        right_layout.addWidget(vn_group)

        # --- Status bar (fixed height to prevent layout jitter) ---
        self._status_label = QLabel(
            "V=0 m/s | Alt=0 m | nz=1.00 | "
            "Sim: 0 Hz | ROM: 0 Hz | GUI: 0 Hz | Tip: 0 mm"
        )
        mono_font = QFont("Menlo", 10)
        mono_font.setStyleHint(QFont.Monospace)
        self._status_label.setFont(mono_font)
        self._status_label.setFixedHeight(20)
        right_layout.addWidget(self._status_label)

        main_layout.addWidget(right_panel)

    def _setup_3d_viewport(self):
        """Initialize PyVista 3D viewport."""
        self._plotter = QtInteractor(self._viewport_frame)
        self._viewport_layout.addWidget(self._plotter.interactor)

        # Build structural mesh (shell elements) with deformation scalars
        self._mesh = build_structural_mesh(self.bdf_model, include_beams=False)
        # Initialize displacement magnitude scalars (all zeros = undeformed)
        self._mesh.point_data['Displacement_Magnitude'] = np.zeros(
            self._mesh.n_points)
        self._mesh_actor = self._plotter.add_mesh(
            self._mesh, scalars='Displacement_Magnitude',
            cmap='jet', clim=[0, 0.01], show_edges=True,
            edge_color='gray', opacity=0.9, name='structure',
            show_scalar_bar=False,
        )

        # Build beam tubes
        beam_mesh = build_beam_tubes(self.bdf_model, n_sides=8)
        if beam_mesh is not None:
            self._beam_mesh = beam_mesh
            self._plotter.add_mesh(
                beam_mesh, color='steelblue', opacity=0.8,
                name='beams',
            )
        else:
            self._beam_mesh = None

        # Save original points for deformation updates
        self._orig_mesh_pts = self._mesh.points.copy()

        # --- Rotor disks: detect from BDF CONM2 motor comments ---
        self._rotor_disks = []
        self._rotor_actors = []
        rotor_hubs = self._detect_rotor_hubs()
        for hub in rotor_hubs:
            disk = pv.Disc(center=hub['center'], normal=hub['normal'],
                           inner=hub['radius'] * 0.08,
                           outer=hub['radius'], r_res=1, c_res=36)
            color = 'lightblue' if hub['type'] == 'lift' else 'lightyellow'
            actor = self._plotter.add_mesh(
                disk, color=color, opacity=0.25,
                name=f"rotor_{hub['name']}", show_edges=False)
            self._rotor_disks.append(disk)
            self._rotor_actors.append(actor)
        if rotor_hubs:
            print(f"       Rotor disks: {len(rotor_hubs)} "
                  f"(radius={rotor_hubs[0]['radius']:.0f} mm)")

        # Camera setup
        self._plotter.camera.position = (15000, -15000, 8000)
        self._plotter.camera.focal_point = (4000, 0, 1000)
        self._plotter.camera.up = (0, 0, 1)
        self._plotter.reset_camera()

        # Add axes
        self._plotter.add_axes()

    def _detect_rotor_hubs(self):
        """Detect rotor hub positions from BDF CONM2 motor comments.

        Scans BDF source lines for "$ motor_N: ... at (x, y, z)" comments.
        Falls back to CONM2 mass clustering if no comments found.
        Returns list of hub dicts with center, normal, radius, name, type.
        """
        hubs = []
        # Strategy 1: parse motor comments from BDF source
        bdf_path = getattr(self.bdf_model, '_source_path', None)
        if bdf_path is None:
            # Try to find path from parser
            bdf_path = getattr(self.bdf_model, 'bdf_path', None)

        if bdf_path:
            try:
                with open(bdf_path) as f:
                    for line in f:
                        s = line.strip()
                        if s.startswith("$ motor_") and " at " in s:
                            name = s.split(":")[0].replace("$ ", "")
                            coord_str = s.split("at")[1].strip().strip("()")
                            vals = coord_str.split(",")
                            if len(vals) == 3:
                                # Coords are in model units (mm after conversion)
                                x, y, z = [float(v.strip()) for v in vals]
                                # Check if already in mm (>100) or m (<100)
                                if abs(x) < 100 and abs(y) < 100:
                                    x *= 1000; y *= 1000; z *= 1000
                                is_pusher = "pusher" in name
                                hubs.append({
                                    'name': name,
                                    'center': [x, y, z],
                                    'normal': [1, 0, 0] if is_pusher else [0, 0, 1],
                                    'radius': 600 if is_pusher else 750,
                                    'type': 'pusher' if is_pusher else 'lift',
                                })
            except Exception:
                pass

        # Strategy 2: if no comments, detect from heavy CONM2 at high Z
        if not hubs:
            for mid, mass_elem in self.bdf_model.masses.items():
                nid = mass_elem.node_id
                if nid not in self.bdf_model.nodes:
                    continue
                if mass_elem.mass < 0.01:
                    continue
                pos = self.bdf_model.nodes[nid].xyz_global
                # Heuristic: motors are at high Z or at extreme Y
                if abs(pos[2]) > 800 or abs(pos[1]) > 3000:
                    hubs.append({
                        'name': f"rotor_{nid}",
                        'center': list(pos),
                        'normal': [0, 0, 1],
                        'radius': 750,
                        'type': 'lift',
                    })

        return hubs

    def _setup_plots(self):
        """Configure pyqtgraph plots."""
        pg.setConfigOptions(antialias=True)

    def _setup_vn_diagram(self):
        """Set up V-n diagram using pyqtgraph."""
        self._vn_plot = pg.PlotWidget(title="V-n Diagram")
        self._vn_plot.setLabel('left', 'Load Factor (nz)')
        self._vn_plot.setLabel('bottom', 'V_EAS (m/s)')
        self._vn_plot.setFixedHeight(200)
        self._vn_layout.addWidget(self._vn_plot)

        # Draw V-n envelope if data available
        if self.vn_data is not None:
            vn = self.vn_data
            # Maneuver envelope
            if hasattr(vn, 'maneuver_curve') and vn.maneuver_curve:
                V_m = [pt[0] for pt in vn.maneuver_curve]
                n_m = [pt[1] for pt in vn.maneuver_curve]
                self._vn_plot.plot(V_m, n_m,
                                   pen=pg.mkPen('b', width=2, style=Qt.SolidLine))
            # Gust envelope positive
            if hasattr(vn, 'gust_curve_pos') and vn.gust_curve_pos:
                V_gp = [pt[0] for pt in vn.gust_curve_pos]
                n_gp = [pt[1] for pt in vn.gust_curve_pos]
                self._vn_plot.plot(V_gp, n_gp,
                                   pen=pg.mkPen('c', width=1, style=Qt.DashLine))
            # Gust envelope negative
            if hasattr(vn, 'gust_curve_neg') and vn.gust_curve_neg:
                V_gn = [pt[0] for pt in vn.gust_curve_neg]
                n_gn = [pt[1] for pt in vn.gust_curve_neg]
                self._vn_plot.plot(V_gn, n_gn,
                                   pen=pg.mkPen('c', width=1, style=Qt.DashLine))

        # Add limit lines
        self._vn_plot.addLine(y=0, pen=pg.mkPen('w', width=0.5, style=Qt.DashLine))
        self._vn_plot.addLine(y=1, pen=pg.mkPen('w', width=0.5, style=Qt.DotLine))

        # Operating point marker (red dot)
        self._vn_marker = pg.ScatterPlotItem(
            size=12, brush=pg.mkBrush('r'), pen=pg.mkPen('w', width=1))
        self._vn_plot.addItem(self._vn_marker)

        # Trail showing recent path
        self._vn_trail = self._vn_plot.plot(
            pen=pg.mkPen('r', width=1, style=Qt.DotLine))

    # -------------------------------------------------------------------
    # Slider callbacks
    # -------------------------------------------------------------------

    def _on_elev_changed(self, val):
        deg = val / 10.0
        self._elev_label.setText(f"Elevator: {deg:.1f}°")
        de, da, dr = self.shared.get_controls()
        self.shared.set_controls(math.radians(deg), da, dr)

    def _on_ail_changed(self, val):
        deg = val / 10.0
        self._ail_label.setText(f"Aileron: {deg:.1f}°")
        de, da, dr = self.shared.get_controls()
        self.shared.set_controls(de, math.radians(deg), dr)

    def _on_rud_changed(self, val):
        deg = val / 10.0
        self._rud_label.setText(f"Rudder: {deg:.1f}°")
        de, da, dr = self.shared.get_controls()
        self.shared.set_controls(de, da, math.radians(deg))

    def _on_scale_changed(self, val):
        self._disp_scale = float(val)
        self._scale_label.setText(f"Deformation Scale: {val}×")

    # -------------------------------------------------------------------
    # Button callbacks
    # -------------------------------------------------------------------

    def _toggle_pause(self):
        self.shared.paused = not self.shared.paused
        self._pause_btn.setText("▶ Resume" if self.shared.paused else "⏸ Pause")

    def _return_to_trim(self):
        """Return to 1g level flight trim state.

        Resets control sliders to zero (= trim deflections) and signals
        the simulation thread to reset its state vector to the initial
        trim condition.  This is equivalent to "teleporting" back to
        steady level flight.
        """
        self._elev_slider.setValue(0)
        self._ail_slider.setValue(0)
        self._rud_slider.setValue(0)
        self.shared.set_controls(0.0, 0.0, 0.0)
        self.shared.reset_to_trim = True

    # -------------------------------------------------------------------
    # Display mode callbacks
    # -------------------------------------------------------------------

    def _on_display_changed(self, idx):
        """Switch between displacement/force/stress colormap."""
        self._display_mode = idx

    def _on_arrow_scale_changed(self, val):
        """Update aero force arrow scale (0=off)."""
        self._arrow_scale_pct = val
        if val == 0:
            self._arrow_label.setText("Aero Arrows (blue): Off")
            if self._aero_arrow_actor is not None:
                try:
                    self._plotter.remove_actor(self._aero_arrow_actor)
                except Exception:
                    pass
                self._aero_arrow_actor = None
        else:
            self._arrow_label.setText(f"Aero Arrows (blue): {val}%")

    def _on_inertia_scale_changed(self, val):
        """Update inertial force arrow scale (0=off)."""
        self._inertia_arrow_scale_pct = val
        if val == 0:
            self._inertia_label.setText("Inertial Arrows (red): Off")
            if self._inertia_arrow_actor is not None:
                try:
                    self._plotter.remove_actor(self._inertia_arrow_actor)
                except Exception:
                    pass
                self._inertia_arrow_actor = None
        else:
            self._inertia_label.setText(f"Inertial Arrows (red): {val}%")

    def _build_aero_arrows(self, snap):
        """Build arrow glyphs showing aero panel forces at panel centers.

        Computes panel-level lift distribution from the current flight
        state and draws arrows at each aero box control point.

        Returns
        -------
        pv.PolyData or None
        """
        alpha = math.radians(snap.get('alpha_deg', 0.0))
        V = snap['V_tas']
        rho = snap['rho']
        de = snap.get('delta_e', 0.0)
        da = snap.get('delta_a', 0.0)
        dr = snap.get('delta_r', 0.0)
        q = 0.5 * rho * V * V * 1e-6  # dynamic pressure (N/mm²)

        if q < 1e-12:
            return None

        # Compute normalwash → panel circulation → pressure → force
        w = (alpha * self._aero_w_alpha
             + de * self._aero_w_elev
             + da * self._aero_w_ail
             + dr * self._aero_w_rud)
        gamma = self._aero_D_inv @ w
        dCp = 2.0 * gamma / self._aero_chords
        panel_force = q * dCp * self._aero_areas  # scalar (N)

        # Force vectors at panel centers
        # Negate: VLM convention gives negative dCp for positive lift,
        # so panel_force * normal points downward.  Negate to get the
        # physical aerodynamic force direction (lift → upward).
        force_vecs = -panel_force[:, None] * self._aero_normals  # (n_boxes, 3)
        mags = np.abs(panel_force)
        max_mag = float(np.max(mags))
        if max_mag < 1e-10:
            return None

        # Filter small forces (< 5% of max)
        keep = mags > (max_mag * 0.05)
        origins = self._aero_centers[keep]
        fv = force_vecs[keep]
        fm = mags[keep]

        # Arrow scale from slider
        user_frac = self._arrow_scale_pct / 100.0
        target_max_len = user_frac * 0.50 * self._bbox_diag
        arrow_scale = target_max_len / max_mag
        lengths = fm * arrow_scale
        min_len = 0.05 * target_max_len
        lengths = np.maximum(lengths, min_len)

        directions = fv / np.linalg.norm(fv, axis=1, keepdims=True)

        pts = pv.PolyData(origins)
        pts['vectors'] = directions * lengths[:, np.newaxis]
        pts['Force_N'] = fm

        arrows = pts.glyph(
            orient='vectors',
            scale='vectors',
            factor=1.0,
            geom=pv.Arrow(
                tip_length=0.25,
                tip_radius=0.12,
                shaft_radius=0.05,
                shaft_resolution=10,
                tip_resolution=10,
            ),
        )
        return arrows

    def _build_inertia_arrows(self, snap):
        """Build arrow glyphs showing inertial (weight) forces at mass nodes.

        Inertial force at each node = m_i × nz × g, pointing downward (-Z).
        For 1g level flight, this shows the weight distribution.

        Returns
        -------
        pv.PolyData or None
        """
        if len(self._mass_node_values) == 0:
            return None

        nz = snap.get('nz', 1.0)
        # Inertial force magnitude at each mass node (N)
        # F_inertia = m * nz * g  (g in mm/s² → force in N when mass in kg)
        # Note: model units are N-mm-sec, mass is in kg-equivalent
        force_mag = self._mass_node_values * nz * self._mass_g_accel  # N

        max_mag = float(np.max(force_mag))
        if max_mag < 1e-10:
            return None

        # Filter small forces (< 5% of max)
        keep = force_mag > (max_mag * 0.05)
        origins = self._mass_node_positions[keep]
        fm = force_mag[keep]

        # All arrows point downward (-Z direction = gravity)
        down = np.zeros((len(fm), 3))
        down[:, 2] = -1.0

        # Arrow scale from slider
        user_frac = self._inertia_arrow_scale_pct / 100.0
        target_max_len = user_frac * 0.50 * self._bbox_diag
        arrow_scale = target_max_len / max_mag
        lengths = fm * arrow_scale
        min_len = 0.05 * target_max_len
        lengths = np.maximum(lengths, min_len)

        pts = pv.PolyData(origins)
        pts['vectors'] = down * lengths[:, np.newaxis]
        pts['Force_N'] = fm

        arrows = pts.glyph(
            orient='vectors',
            scale='vectors',
            factor=1.0,
            geom=pv.Arrow(
                tip_length=0.25,
                tip_radius=0.12,
                shaft_radius=0.05,
                shaft_resolution=10,
                tip_resolution=10,
            ),
        )
        return arrows

    # -------------------------------------------------------------------
    # Keyboard handling
    # -------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        step_e = 10   # 1.0 degree steps
        step_a = 10
        step_r = 10

        if key == Qt.Key_Up:
            self._elev_slider.setValue(
                max(self._elev_slider.value() - step_e, self._elev_slider.minimum()))
        elif key == Qt.Key_Down:
            self._elev_slider.setValue(
                min(self._elev_slider.value() + step_e, self._elev_slider.maximum()))
        elif key == Qt.Key_Left:
            self._ail_slider.setValue(
                max(self._ail_slider.value() - step_a, self._ail_slider.minimum()))
        elif key == Qt.Key_Right:
            self._ail_slider.setValue(
                min(self._ail_slider.value() + step_a, self._ail_slider.maximum()))
        elif key == Qt.Key_Comma:
            self._rud_slider.setValue(
                max(self._rud_slider.value() - step_r, self._rud_slider.minimum()))
        elif key == Qt.Key_Period:
            self._rud_slider.setValue(
                min(self._rud_slider.value() + step_r, self._rud_slider.maximum()))
        elif key == Qt.Key_Space:
            self._toggle_pause()
        elif key == Qt.Key_R:
            self._return_to_trim()
        else:
            super().keyPressEvent(event)

    # -------------------------------------------------------------------
    # Timer callback — 20 Hz GUI update
    # -------------------------------------------------------------------

    def _on_timer(self):
        snap = self.shared.get_snapshot()

        # GUI FPS
        self._gui_frame_count += 1
        now = time.perf_counter()
        if now - self._gui_fps_time >= 0.5:
            self._gui_fps = self._gui_frame_count / (now - self._gui_fps_time)
            self._gui_frame_count = 0
            self._gui_fps_time = now

        # --- Update time history ---
        t = snap['sim_time']
        self._t_hist = np.roll(self._t_hist, -1)
        self._t_hist[-1] = t
        self._nz_hist = np.roll(self._nz_hist, -1)
        self._nz_hist[-1] = snap['nz']
        self._alpha_hist = np.roll(self._alpha_hist, -1)
        self._alpha_hist[-1] = snap['alpha_deg']
        self._beta_hist = np.roll(self._beta_hist, -1)
        self._beta_hist[-1] = snap['beta_deg']
        self._p_hist = np.roll(self._p_hist, -1)
        self._p_hist[-1] = math.degrees(snap['p'])
        self._q_hist = np.roll(self._q_hist, -1)
        self._q_hist[-1] = math.degrees(snap['q'])
        self._r_hist = np.roll(self._r_hist, -1)
        self._r_hist[-1] = math.degrees(snap['r'])

        # Update plot curves
        self._curve_nz.setData(self._t_hist, self._nz_hist)
        self._curve_alpha.setData(self._t_hist, self._alpha_hist)
        self._curve_beta.setData(self._t_hist, self._beta_hist)
        self._curve_p.setData(self._t_hist, self._p_hist)
        self._curve_q.setData(self._t_hist, self._q_hist)
        self._curve_r.setData(self._t_hist, self._r_hist)

        # Auto-range time axis to recent window
        if t > 10.0:
            self._plot_nz.setXRange(t - 10.0, t)
            self._plot_rates.setXRange(t - 10.0, t)

        # --- Update V-n operating point ---
        V_eas = snap['V_tas'] * math.sqrt(snap['rho'] / 1.225)
        nz = snap['nz']
        self._vn_marker.setData([V_eas], [nz])

        # Trail (last 60 points ≈ 3s)
        self._vn_trail_V.append(V_eas)
        self._vn_trail_n.append(nz)
        if len(self._vn_trail_V) > 60:
            self._vn_trail_V = self._vn_trail_V[-60:]
            self._vn_trail_n = self._vn_trail_n[-60:]
        self._vn_trail.setData(self._vn_trail_V, self._vn_trail_n)

        # --- Update 3D mesh deformation ---
        # Synthetic bending only — FEA displacement is too small at spars
        # (~0.005 mm tip at 2.5g) and creates local spikes at spline nodes
        # when amplified. Synthetic bending gives smooth, physically-scaled
        # deformation driven directly by flight state (nz, δa, β).
        disp_arr = snap['disp_array']
        force_arr = snap['force_array']
        if disp_arr is not None:
            new_pts = self._orig_mesh_pts.copy()

            delta_nz = snap['nz'] - 1.0   # excess g above level flight
            beta_rad = math.radians(snap.get('beta_deg', 0.0))
            delta_a = snap.get('delta_a', 0.0)
            delta_e = snap.get('delta_e', 0.0)

            # Wing + HTP: z-deflection from nz (cubic cantilever shape)
            synth_dz = delta_nz * self._synth_bend_z  # (n_total,)
            new_pts[:, 2] += synth_dz

            # Aileron differential bending: opposite sign for L/R wing
            if abs(delta_a) > 1e-6:
                ail_scale = delta_a * 15.0  # amplify aileron effect
                ail_dz = ail_scale * self._synth_bend_z
                new_pts[self._lwing_mask, 2] -= ail_dz[self._lwing_mask]
                new_pts[self._rwing_mask, 2] += ail_dz[self._rwing_mask]

            # VTP: y-deflection from sideslip
            if abs(beta_rad) > 0.01:
                new_pts[:, 1] += beta_rad * self._synth_bend_y

            # --- Control surface deflection ---
            # Elevator: TE nodes rotate about hinge line
            # dz = arm * sin(delta_e), positive delta_e = TE down
            if abs(delta_e) > 1e-4:
                elev_dz = self._cs_arm_elev * math.sin(delta_e)
                new_pts[:, 2] -= elev_dz  # negative = TE moves down for +delta_e

            # Aileron: right TE down for +delta_a, left TE up
            if abs(delta_a) > 1e-4:
                new_pts[:, 2] -= self._cs_arm_ail_r * math.sin(delta_a)
                new_pts[:, 2] += self._cs_arm_ail_l * math.sin(delta_a)

            self._mesh.points = new_pts

            # --- Colormap based on display mode ---
            if self._display_mode == 0:
                # Displacement mode: synthetic bending magnitude
                synth_mag = np.abs(synth_dz)
                self._mesh.point_data['Displacement_Magnitude'] = synth_mag
                wing_synth = synth_mag[self._wing_mask]
                max_mag = float(np.max(wing_synth)) if len(wing_synth) > 0 else 0.0
                max_mag = max(max_mag, 1.0)
                try:
                    self._mesh_actor.mapper.scalar_range = [0, max_mag]
                except Exception:
                    pass

            elif force_arr is not None and self._display_mode in (1, 2):
                # Force mode: map ROM force data → mesh point scalars
                n_map = len(self._rom_mesh_idx)
                self._force_scalars[:] = 0.0

                if self._display_mode == 1:
                    # Fz only (bending load, component 2)
                    fz_vals = force_arr[:n_map, 2]
                    self._force_scalars[self._rom_mesh_idx] = np.abs(fz_vals)
                else:
                    # |F| total magnitude
                    f_mag = np.linalg.norm(force_arr[:n_map, :3], axis=1)
                    self._force_scalars[self._rom_mesh_idx] = f_mag

                self._mesh.point_data['Displacement_Magnitude'] = \
                    self._force_scalars

                # Color range from element-connected nodes only
                elem_forces = self._force_scalars[self._elem_mask]
                max_f = float(np.max(elem_forces)) if len(elem_forces) > 0 else 1.0
                max_f = max(max_f, 1.0)
                try:
                    self._mesh_actor.mapper.scalar_range = [0, max_f]
                except Exception:
                    pass

            elif self._display_mode == 3:
                # Stress mode: von Mises stress from ROM
                stress_vm = snap.get('stress_vm')
                if stress_vm is not None:
                    n_map = len(self._rom_mesh_idx)
                    self._force_scalars[:] = 0.0
                    n_use = min(n_map, len(stress_vm))
                    self._force_scalars[self._rom_mesh_idx[:n_use]] = \
                        stress_vm[:n_use]

                    self._mesh.point_data['Displacement_Magnitude'] = \
                        self._force_scalars

                    elem_stress = self._force_scalars[self._elem_mask]
                    max_s = float(np.max(elem_stress)) if len(elem_stress) > 0 else 1.0
                    max_s = max(max_s, 1.0)
                    try:
                        self._mesh_actor.mapper.scalar_range = [0, max_s]
                    except Exception:
                        pass

            # --- Aero panel force arrows (blue) ---
            if self._arrow_scale_pct > 0:
                # Remove previous aero arrows
                if self._aero_arrow_actor is not None:
                    try:
                        self._plotter.remove_actor(self._aero_arrow_actor)
                    except Exception:
                        pass
                    self._aero_arrow_actor = None

                # Build aero panel arrows (blue — lift distribution)
                aero_arrows = self._build_aero_arrows(snap)
                if aero_arrows is not None:
                    try:
                        self._aero_arrow_actor = self._plotter.add_mesh(
                            aero_arrows, color='dodgerblue',
                            opacity=0.85, name='aero_arrows',
                            show_scalar_bar=False,
                        )
                    except Exception:
                        pass
            elif self._aero_arrow_actor is not None:
                try:
                    self._plotter.remove_actor(self._aero_arrow_actor)
                except Exception:
                    pass
                self._aero_arrow_actor = None

            # --- Inertial force arrows (red) ---
            if self._inertia_arrow_scale_pct > 0:
                # Remove previous inertial arrows
                if self._inertia_arrow_actor is not None:
                    try:
                        self._plotter.remove_actor(self._inertia_arrow_actor)
                    except Exception:
                        pass
                    self._inertia_arrow_actor = None

                inertia_arrows = self._build_inertia_arrows(snap)
                if inertia_arrows is not None:
                    try:
                        self._inertia_arrow_actor = self._plotter.add_mesh(
                            inertia_arrows, color='red',
                            opacity=0.85, name='inertia_arrows',
                            show_scalar_bar=False,
                        )
                    except Exception:
                        pass
            elif self._inertia_arrow_actor is not None:
                try:
                    self._plotter.remove_actor(self._inertia_arrow_actor)
                except Exception:
                    pass
                self._inertia_arrow_actor = None

        # Force plotter render
        try:
            self._plotter.render()
        except Exception:
            pass

        # --- Update status bar ---
        tip_dz = abs(snap['nz'] - 1.0) * float(np.max(self._synth_bend_z))

        # Show force/stress info when in force/stress mode
        force_info = ""
        if self._display_mode in (1, 2) and force_arr is not None:
            n_map = len(self._rom_mesh_idx)
            if self._display_mode == 1:
                max_fz = float(np.max(np.abs(force_arr[:n_map, 2])))
                force_info = f" | Fz:{max_fz:.0f}N"
            else:
                max_f = float(np.max(np.linalg.norm(
                    force_arr[:n_map, :3], axis=1)))
                force_info = f" | |F|:{max_f:.0f}N"
        elif self._display_mode == 3:
            stress_vm = snap.get('stress_vm')
            if stress_vm is not None:
                max_vm = float(np.max(stress_vm))
                force_info = f" | σ_vm:{max_vm:.1f}MPa"

        # Fixed-width format to prevent layout jitter from text length changes
        status = (
            f"V={snap['V_tas']:6.1f} m/s "
            f"nz={snap['nz']:+5.2f} "
            f"a={snap['alpha_deg']:+5.1f} "
            f"b={snap['beta_deg']:+5.1f} "
            f"Sim:{snap['sim_fps']:4.0f}Hz "
            f"ROM:{snap['rom_fps']:3.0f}Hz "
            f"GUI:{self._gui_fps:3.0f}Hz "
            f"Tip:{tip_dz:5.0f}mm"
        )
        if force_info:
            status += force_info
        if snap['paused']:
            status += " [PAUSED]"
        self._status_label.setText(status)

    # -------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------

    def closeEvent(self, event):
        self.shared.running = False
        self._timer.stop()
        self._sim_thread.wait(2000)
        self._plotter.close()
        super().closeEvent(event)
