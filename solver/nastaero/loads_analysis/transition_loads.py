"""Transition phase loads — quasi-steady SOL 144 sequence + dynamic correction.

Implements the dissertation Chapter 5 algorithm: discretize a transition
maneuver into N_t time points, solve quasi-steady SOL 144 trim at each,
then add dynamic correction term using finite-difference inertial loads.

Status
------
Dissertation v0.2 [PLAN]: This module formalizes the v0.1 prototype
algorithm. Main work remaining:
  1. Integration with vtol_load_case_matrix.VTOLLoadCaseMatrix
  2. Validation against transient FE analysis (SOL 109/112)
  3. Coupling with rotor BEMT for mixed lift contribution
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import numpy as np


# ============================================================
# Lift sharing function (Ch 5 §5.2.1)
# ============================================================

@dataclass
class LiftSharingModel:
    """Wing lift fraction lambda(V) model.

    Default: logistic fit to NASA L+C data (Johnson 2018):
        lambda(V) = 1 / [1 + exp(-k * (V - V_50))]

    with V_50 = 28 m/s, k = 0.18 s/m.

    Source tag: [CALC] for fit values; [LIT] for NASA L+C data points.

    Attributes
    ----------
    V_50 : float
        Balance speed where lambda = 0.5 (m/s).
    k : float
        Hand-off slope (s/m).
    """
    V_50: float = 28.0
    k: float = 0.18

    def lambda_of(self, V: float) -> float:
        """lambda(V) — wing lift fraction at forward speed V."""
        return 1.0 / (1.0 + np.exp(-self.k * (V - self.V_50)))

    def lambda_dot_of(self, V: float, V_dot: float) -> float:
        """d(lambda)/dt = d(lambda)/dV * dV/dt."""
        lam = self.lambda_of(V)
        dlam_dV = self.k * lam * (1.0 - lam)
        return dlam_dV * V_dot

    def fit_residuals(
        self,
        V_data: np.ndarray,
        lambda_data: np.ndarray,
    ) -> dict:
        """Compute fitting residuals against given data points.

        NOTE: This method does NOT perform fitting; it only evaluates
        residuals using the model's *current* (V_50, k) parameters.
        To perform an actual nonlinear least-squares fit, use the
        classmethod ``fit_to_data()``.

        Parameters
        ----------
        V_data, lambda_data : np.ndarray
            Reference data (e.g., NASA TM trim sweep).

        Returns
        -------
        dict with 'residuals', 'rms', 'max_abs'.
        """
        lam_fit = np.array([self.lambda_of(v) for v in V_data])
        residuals = lambda_data - lam_fit
        return {
            "V": V_data,
            "lambda_data": lambda_data,
            "lambda_fit": lam_fit,
            "residuals": residuals,
            "rms": float(np.sqrt(np.mean(residuals ** 2))),
            "max_abs": float(np.max(np.abs(residuals))),
        }

    @classmethod
    def fit_to_data(
        cls,
        V_data: np.ndarray,
        lambda_data: np.ndarray,
        V_50_init: float = 28.0,
        k_init: float = 0.18,
    ) -> "LiftSharingModel":
        """Fit a LiftSharingModel to (V, λ) data via nonlinear least-squares.

        Solves min over (V_50, k):
            sum_i [ λ_data_i - 1/(1+exp(-k(V_data_i - V_50))) ]^2

        Parameters
        ----------
        V_data, lambda_data : np.ndarray
            Data points to fit.
        V_50_init, k_init : float
            Initial guesses for the optimizer.

        Returns
        -------
        LiftSharingModel with optimal (V_50, k).

        Raises
        ------
        ImportError
            If scipy is not available.
        """
        try:
            from scipy.optimize import least_squares
        except ImportError as e:
            raise ImportError(
                "fit_to_data requires scipy.optimize.least_squares. "
                "Install scipy or use the model with manual (V_50, k)."
            ) from e

        V_data = np.asarray(V_data, dtype=float)
        lambda_data = np.asarray(lambda_data, dtype=float)

        def residuals(params):
            V_50, k = params
            lam_pred = 1.0 / (1.0 + np.exp(-k * (V_data - V_50)))
            return lam_pred - lambda_data

        result = least_squares(
            residuals,
            x0=[V_50_init, k_init],
            method="lm",
            xtol=1e-10, ftol=1e-10,
        )
        V_50_opt, k_opt = result.x
        return cls(V_50=float(V_50_opt), k=float(k_opt))


# ============================================================
# Quasi-steady time discretization
# ============================================================

@dataclass
class TransitionTimeline:
    """Time discretization of a transition maneuver.

    Default: 30 s hover→cruise transition with 60 quasi-steady time points
    (Delta_t = 0.5 s). Convergence study in Ch 5 §5.3.4 confirms < 1%
    change at N_t = 60 vs 120.

    Parameters
    ----------
    t_start, t_end : float
        Maneuver start/end times (s).
    N_t : int
        Number of quasi-steady time points.
    V_cruise : float
        Cruise speed at t_end (m/s).
    V_ramp_type : str
        "linear" or future {"power_minimization"} from path optimization.
    """
    t_start: float = 0.0
    t_end: float = 30.0
    N_t: int = 60
    V_cruise: float = 67.0
    V_ramp_type: str = "linear"

    def __post_init__(self):
        self.t_grid = np.linspace(self.t_start, self.t_end, self.N_t)
        self.dt = self.t_grid[1] - self.t_grid[0]

    def V_at(self, t: float) -> float:
        """V(t) — forward speed at time t."""
        if self.V_ramp_type == "linear":
            return self.V_cruise * np.minimum(t / self.t_end, 1.0)
        raise NotImplementedError(f"V_ramp_type='{self.V_ramp_type}'")

    def V_dot_at(self, t: float) -> float:
        """dV/dt at time t."""
        if self.V_ramp_type == "linear":
            return self.V_cruise / self.t_end if t < self.t_end else 0.0
        raise NotImplementedError(f"V_ramp_type='{self.V_ramp_type}'")


# ============================================================
# Dynamic correction (Ch 5 §5.3.3)
# ============================================================

@dataclass
class DynamicCorrection:
    """Compute dynamic correction loads from quasi-steady displacement sequence.

    F_dyn_i = -M @ u_ddot_i^qs - C @ u_dot_i^qs

    where u_dot, u_ddot are obtained by central finite difference on the
    quasi-steady displacement sequence {u_i^qs}.

    Caveat (Ch 5 §5.3.3 limitation note): This is a post-processing
    estimate; rigorous transient structural analysis (SOL 109/112)
    validation is v0.2 [PLAN] work.

    Parameters
    ----------
    M : sparse or dense matrix
        Mass matrix (N_dof x N_dof).
    C : sparse or dense matrix
        Damping matrix. Default = 0 (undamped); typically Rayleigh.
    """
    M: np.ndarray  # could be scipy sparse
    C: Optional[np.ndarray] = None

    def compute(
        self,
        u_qs_sequence: np.ndarray,
        t_grid: np.ndarray,
    ) -> np.ndarray:
        """Compute F_dyn at each time point.

        Parameters
        ----------
        u_qs_sequence : np.ndarray of shape (N_t, N_dof)
            Quasi-steady displacement at each time point.
        t_grid : np.ndarray of shape (N_t,)
            Time grid.

        Returns
        -------
        np.ndarray of shape (N_t, N_dof)
            Dynamic correction force at each time point.
        """
        N_t, N_dof = u_qs_sequence.shape
        if len(t_grid) != N_t:
            raise ValueError("t_grid and u_qs_sequence length mismatch")

        # Central finite difference for u_dot, u_ddot
        # (forward/backward at endpoints)
        u_dot = np.zeros_like(u_qs_sequence)
        u_ddot = np.zeros_like(u_qs_sequence)

        # Interior points: central difference
        for i in range(1, N_t - 1):
            dt_pm = t_grid[i + 1] - t_grid[i - 1]
            u_dot[i] = (u_qs_sequence[i + 1] - u_qs_sequence[i - 1]) / dt_pm
            dt_sq = ((t_grid[i + 1] - t_grid[i]) *
                     (t_grid[i] - t_grid[i - 1]))
            u_ddot[i] = ((u_qs_sequence[i + 1] - 2 * u_qs_sequence[i]
                          + u_qs_sequence[i - 1]) / dt_sq)

        # Endpoints: forward/backward difference
        u_dot[0] = (u_qs_sequence[1] - u_qs_sequence[0]) / (t_grid[1] - t_grid[0])
        u_dot[-1] = (u_qs_sequence[-1] - u_qs_sequence[-2]) / (t_grid[-1] - t_grid[-2])
        u_ddot[0] = u_ddot[1]   # Use neighbor for second derivative
        u_ddot[-1] = u_ddot[-2]

        # F_dyn = -M @ u_ddot - C @ u_dot
        F_dyn = np.zeros_like(u_qs_sequence)
        for i in range(N_t):
            F_dyn[i] = -self.M @ u_ddot[i]
            if self.C is not None:
                F_dyn[i] -= self.C @ u_dot[i]

        return F_dyn

    def convergence_check(
        self,
        u_qs_N: np.ndarray, t_grid_N: np.ndarray,
        u_qs_2N: np.ndarray, t_grid_2N: np.ndarray,
        tol: float = 0.01,
    ) -> dict:
        """Verify dynamic correction convergence at refined time grid.

        Returns dict with 'converged', 'max_relative_error'.
        """
        F_N = self.compute(u_qs_N, t_grid_N)
        F_2N = self.compute(u_qs_2N, t_grid_2N)

        # Resample F_2N at coarse grid points
        F_2N_at_N = F_2N[::2]   # Take every other point

        if F_N.shape != F_2N_at_N.shape:
            raise ValueError("Coarse and refined grids not 2:1")

        norm_diff = np.linalg.norm(F_N - F_2N_at_N)
        norm_total = np.linalg.norm(F_N) + 1e-12
        rel_err = norm_diff / norm_total

        return {
            "converged": rel_err < tol,
            "max_relative_error": float(rel_err),
            "tol": tol,
        }


# ============================================================
# Transition critical metric (12-D state, Ch 5 §5.4)
# ============================================================

# Index for s^TR = [n_z, alpha, beta, V, p, q, r, de, da, dr, lambda, dlambda/dt]
TRANSITION_STATE_NAMES = [
    "nz", "alpha", "beta", "V", "p", "q", "r",
    "delta_e", "delta_a", "delta_r",
    "lambda", "dlambda_dt",
]
TRANSITION_STATE_DIM = 12

# Default weights (Ch 5 §5.4.3): higher for nz and lambda dynamics
TRANSITION_DEFAULT_WEIGHTS = np.array([
    3.0,  # nz
    1.5, 1.5,  # alpha, beta
    0.5,  # V
    2.0, 2.0, 2.0,  # p, q, r
    1.0, 1.0, 1.0,  # de, da, dr
    2.5, 2.5,  # lambda, dlambda/dt
])


def transition_distance(
    s_i: np.ndarray, s_j: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Weighted Euclidean distance between two 12-D transition states.

    d(s_i, s_j) = sqrt(sum_k w_k (s_i,k - s_j,k)^2)

    Used for farthest-point sampling of critical time hacks.
    """
    if weights is None:
        weights = TRANSITION_DEFAULT_WEIGHTS
    if len(s_i) != TRANSITION_STATE_DIM or len(s_j) != TRANSITION_STATE_DIM:
        raise ValueError(f"State must be {TRANSITION_STATE_DIM}-D")
    diff = s_i - s_j
    return float(np.sqrt(np.sum(weights * diff ** 2)))


def farthest_point_sampling(
    candidates: np.ndarray,
    n_select: int,
    weights: Optional[np.ndarray] = None,
    seed_idx: int = 0,
) -> List[int]:
    """Greedy farthest-point sampling in weighted 12-D state space.

    Parameters
    ----------
    candidates : np.ndarray of shape (N, 12)
        Candidate critical points.
    n_select : int
        Number of unique cases to select.
    weights : np.ndarray of shape (12,), optional
        Per-component weights.
    seed_idx : int
        Index of the first selected point (default 0).

    Returns
    -------
    list of int
        Selected indices into `candidates`.
    """
    if weights is None:
        weights = TRANSITION_DEFAULT_WEIGHTS
    N = candidates.shape[0]
    if n_select > N:
        return list(range(N))

    selected = [seed_idx]
    min_dist = np.full(N, np.inf)

    for _ in range(n_select - 1):
        # Update min distance to selected set
        for i in range(N):
            d = transition_distance(candidates[i], candidates[selected[-1]], weights)
            if d < min_dist[i]:
                min_dist[i] = d

        # Pick farthest
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)

    return selected


# ============================================================
# Smoke test
# ============================================================

def _smoke_test():
    """Verify module loads and core algorithms run."""
    print("Transition loads module smoke test:")

    # Test LiftSharingModel
    lsm = LiftSharingModel()
    print(f"  lambda(V=0)  = {lsm.lambda_of(0.0):.3f}")
    print(f"  lambda(V=28) = {lsm.lambda_of(28.0):.3f}")
    print(f"  lambda(V=67) = {lsm.lambda_of(67.0):.3f}")

    # Test residuals against NASA-like data
    V_data = np.array([0.0, 28.0, 50.0, 67.0])
    lam_data = np.array([0.02, 0.50, 0.92, 0.99])
    res = lsm.fit_residuals(V_data, lam_data)
    print(f"  RMS residual = {res['rms']:.4f}   (target: ~0.018 [CALC])")

    # Test transition timeline
    timeline = TransitionTimeline(N_t=60)
    print(f"\n  Timeline: N_t={timeline.N_t}, dt={timeline.dt:.3f} s")
    print(f"  V(t=8)  = {timeline.V_at(8.0):.1f} m/s")
    print(f"  V(t=15) = {timeline.V_at(15.0):.1f} m/s")

    # Test farthest-point sampling
    rng = np.random.default_rng(42)
    candidates = rng.standard_normal((50, 12))
    selected = farthest_point_sampling(candidates, n_select=12)
    print(f"\n  FPS: 50 candidates → {len(selected)} unique cases")
    print(f"  Selected indices: {selected[:5]}...")


if __name__ == "__main__":
    _smoke_test()
