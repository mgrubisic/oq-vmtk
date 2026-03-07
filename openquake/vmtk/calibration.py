"""
MDOF Calibration Module
=======================

Calibration of Multi-Degree-of-Freedom (MDOF) storey force-deformation
relationships from Single-Degree-of-Freedom (SDOF) spectral capacity parameters.

Two assumptions are fixed for all systems and building heights:

    Uniform ductility  : all storeys share the same displacement backbone.
                         Every storey drift equals the first-storey drift
                         (Sd × Γ × φ[0]), so only storey forces vary with
                         height via the first-mode shear ratio V_i / V_base.

    Stiffness grouping : adjacent storeys are paired (group_size = 2),
                         reducing the number of independent spring stiffnesses
                         to ceil(nst / 2).

No higher-mode correction is applied.  Only the first mode is used for the
SDOF-to-MDOF transformation.

Theory
------
The transformation follows the N2 / Capacity Spectrum method:

    Base shear:         V_base = Sa × M_eff
    First-storey drift: δ₁    = Sd × Γ × φ[0]
    Storey shear i:     V_i   = V_base × SR_i      SR_i = Σ_{j≥i}(m_j φ_j) / Σ(m_j φ_j)
    Storey drift i:     δ_i   = δ₁                 (uniform ductility)

References
----------
.. [1] Fajfar, P. (1999). "Capacity spectrum method based on inelastic demand
       spectra." Earthquake Engng Struct Dyn.

.. [2] Priestley, M.J.N., Calvi, G.M., and Kowalsky, M.J. (2007).
       "Displacement-Based Seismic Design of Structures." IUSS Press.
"""

import numpy as np
from scipy.linalg import eigh
from typing import Tuple, Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass
import warnings
from openquake.vmtk.modeller import modeller as _modeller

# Constants
G = 9.81  # Gravitational acceleration (m/s²)

# Building height categories
LOW_RISE_MAX = 3      # 1-3 storeys
MID_RISE_MAX = 9      # 4-9 storeys
# HIGH_RISE: 10+ storeys


# =============================================================================
# ENUMERATIONS
# =============================================================================

class BuildingCategory(Enum):
    """Building height classification."""
    LOW_RISE = "low-rise"      # 1-3 storeys
    MID_RISE = "mid-rise"      # 4-9 storeys
    HIGH_RISE = "high-rise"    # 10+ storeys


@dataclass
class CalibrationConfig:
    """
    Configuration for MDOF calibration.

    All assumptions are fixed:
    - Uniform ductility  : all storeys share the same drift backbone.
    - Stiffness grouping : group_size = 2 for all systems.
    - Stiffness decay    : k = 1.0 - 0.30*h, min 0.50 for all systems.
    - Roof mass factor   : 0.75 for all systems.
    - Higher modes       : not corrected; only the first mode is used.
    """
    roof_mass_factor:     float = 0.75
    soft_storey_factor:   float = 0.50
    stiffness_group_size: int   = 2
    period_tolerance:     float = 0.02
    validate_results:     bool  = True
    verbose:              bool  = False


# =============================================================================
# BUILDING CLASSIFICATION
# =============================================================================

def classify_building(nst: int) -> BuildingCategory:
    """Classify building by number of storeys."""
    if nst <= LOW_RISE_MAX:
        return BuildingCategory.LOW_RISE
    elif nst <= MID_RISE_MAX:
        return BuildingCategory.MID_RISE
    else:
        return BuildingCategory.HIGH_RISE



def build_mass_matrix(nst: int,
                      roof_mass_factor: float = 0.75,
                      mass_profile: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Build diagonal mass matrix normalised to a total mass of 1.0.

    The SDOF capacity curve is defined for a unit-mass oscillator (1 tonne),
    so the MDOF mass matrix must also sum to 1.0 for the modal transformation
    to remain consistent:

        V_base = Sa × M_eff    where M_eff = (φᵀ M 1)² / (φᵀ M φ)

    If M_total ≠ 1 then M_eff is scaled by M_total and the base shear
    distribution is wrong before the period correction absorbs it.

    Mass distribution
    -----------------
    All intermediate floors carry an equal share; the roof carries a reduced
    fraction controlled by *roof_mass_factor* (< 1.0):

        m_floor = 1.0 / (nst - 1 + roof_mass_factor)
        m_roof  = m_floor × roof_mass_factor

    so that  Σ m_i = (nst-1)×m_floor + m_roof = 1.0  exactly.

    For nst = 1 the single floor mass is 1.0 (SDOF, no roof reduction).

    Parameters
    ----------
    nst : int
        Number of storeys.
    roof_mass_factor : float
        Ratio of roof mass to typical floor mass (default 0.75).
        Use 0.70 for wall systems (lighter roof parapet).
    mass_profile : np.ndarray, optional
        If provided, used directly as the diagonal (must sum to 1.0 or be
        pre-scaled; length must equal nst).
    """
    if mass_profile is not None:
        return np.diag(np.asarray(mass_profile, dtype=float))
    if nst == 1:
        return np.array([[1.0]])
    m_floor = 1.0 / (nst - 1 + roof_mass_factor)
    masses = np.full(nst, m_floor)
    masses[-1] = m_floor * roof_mass_factor
    return np.diag(masses)


def _storey_stiffness_profile(nst: int, group_size: int = 2) -> np.ndarray:
    """
    Relative inter-storey stiffness profile: k = 1.0 - 0.30*h, min 0.50.
    Sections change every group_size storeys (fixed at 2).
    h = normalised height of the group centroid (0 at ground, 1 at roof).
    """
    n_groups = int(np.ceil(nst / group_size))
    k_groups = np.ones(n_groups)
    for g in range(n_groups):
        i_mid = g * group_size + (group_size - 1) / 2.0
        h = i_mid / max(nst - 1, 1)
        k_groups[g] = max(1.0 - 0.30 * h, 0.50)
    return np.array([k_groups[min(i // group_size, n_groups - 1)]
                     for i in range(nst)])


def build_stiffness_matrix(nst: int,
                           isSOS: bool = False,
                           soft_storey_factor: float = 0.50,
                           stiffness_profile: Optional[np.ndarray] = None,
                           group_size: int = 2) -> np.ndarray:
    """
    Build tri-diagonal lateral stiffness matrix.

    Uses a uniform decay profile k = 1.0 - 0.30*h (min 0.50) with sections
    grouped every 2 storeys.  Pass stiffness_profile to override entirely.
    If isSOS is True, the ground-floor spring is reduced by soft_storey_factor.
    """
    if stiffness_profile is not None:
        k = np.asarray(stiffness_profile, dtype=float)
    else:
        k = _storey_stiffness_profile(nst, group_size)
        if isSOS:
            k[0] *= soft_storey_factor

    K = np.zeros((nst, nst))
    for i in range(nst):
        K[i, i] = k[i] + (k[i + 1] if i + 1 < nst else 0.0)
        if i > 0:
            K[i, i - 1] = -k[i]
            K[i - 1, i] = -k[i]
    return K



# =============================================================================
# MODAL ANALYSIS
# =============================================================================

def compute_modal_properties(M: np.ndarray,
                              K: np.ndarray,
                              num_modes: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Compute modal properties from mass and stiffness matrices.

    Parameters
    ----------
    M : np.ndarray
        Mass matrix (nst × nst).
    K : np.ndarray
        Stiffness matrix (nst × nst).
    num_modes : int, optional
        Number of modes to compute. Default: all modes.

    Returns
    -------
    modal_props : dict
        Dictionary containing:
        - 'eigenvalues': ω² for each mode
        - 'frequencies': ω (rad/s) for each mode
        - 'periods': T (s) for each mode
        - 'mode_shapes': φ matrix (nst × num_modes), each column is a mode
        - 'participation_factors': Γ for each mode
        - 'effective_masses': M* for each mode
        - 'mass_participation_ratios': MPR for each mode
        - 'cumulative_mass_participation': Cumulative MPR
    """
    nst = M.shape[0]
    if num_modes is None:
        num_modes = nst
    num_modes = min(num_modes, nst)

    # Solve generalized eigenvalue problem: K φ = ω² M φ
    eigenvalues, eigenvectors = eigh(K, M)

    # Sort by eigenvalue (should already be sorted, but ensure)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take only requested modes
    eigenvalues = eigenvalues[:num_modes]
    eigenvectors = eigenvectors[:, :num_modes]

    # Normalize mode shapes: roof displacement = 1.0; first non-zero sign positive
    mode_shapes = np.zeros_like(eigenvectors)
    for i in range(num_modes):
        phi = eigenvectors[:, i]
        # Normalise to roof = 1
        if abs(phi[-1]) > 1e-14:
            phi = phi / phi[-1]
        else:
            max_val = np.max(np.abs(phi))
            if max_val > 1e-14:
                phi = phi / max_val
        # Ensure the first non-zero entry is positive (sign convention)
        for comp in phi:
            if abs(comp) > 1e-10:
                if comp < 0:
                    phi = -phi
                break
        mode_shapes[:, i] = phi

    # Compute modal properties
    ones = np.ones(nst)
    M_total = np.trace(M)

    frequencies = np.sqrt(np.maximum(eigenvalues, 1e-10))
    periods = 2 * np.pi / frequencies

    participation_factors = np.zeros(num_modes)
    effective_masses = np.zeros(num_modes)
    mass_participation_ratios = np.zeros(num_modes)

    for i in range(num_modes):
        phi = mode_shapes[:, i]
        L_n = phi @ M @ ones      # Modal excitation factor
        M_n = phi @ M @ phi       # Generalized mass

        participation_factors[i] = L_n / M_n
        effective_masses[i] = L_n ** 2 / M_n
        mass_participation_ratios[i] = effective_masses[i] / M_total

    cumulative_mpr = np.cumsum(mass_participation_ratios)

    return {
        'eigenvalues': eigenvalues,
        'frequencies': frequencies,
        'periods': periods,
        'mode_shapes': mode_shapes,
        'participation_factors': participation_factors,
        'effective_masses': effective_masses,
        'mass_participation_ratios': mass_participation_ratios,
        'cumulative_mass_participation': cumulative_mpr,
        'M_total': M_total
    }


# =============================================================================
# MODAL DISTRIBUTION
# =============================================================================


def compute_storey_distribution_ratios(phi: np.ndarray,
                                        M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute shear and drift distribution ratios from mode shape.

    Returns
    -------
    shear_ratios : np.ndarray
        V_i / V_base for each storey  (Σm_j φ_j from storey i up / Σ all).
    drift_ratios : np.ndarray
        Δφ_i / Δφ_1  for each storey, where Δφ_i = φ[i] - φ[i-1]
        and Δφ_1 = φ[0] (ground-to-first-floor inter-storey mode-shape increment).
    """
    nst = len(phi)
    m_diag = np.diag(M)

    # Shear ratios: V_i / V_base = Σ_{j>=i}(m_j φ_j) / Σ_j(m_j φ_j)
    total = np.sum(m_diag * phi)
    if abs(total) < 1e-14:
        shear_ratios = np.ones(nst)
    else:
        shear_ratios = np.array([np.sum(m_diag[i:] * phi[i:]) / total for i in range(nst)])

    # Drift ratios: Δφ_i / Δφ_1
    # Δφ_0 (ground to floor 1) = phi[0] - 0 = phi[0]
    # Δφ_i (floor i to floor i+1) = phi[i] - phi[i-1]  for i >= 1
    delta_phi = np.empty(nst)
    delta_phi[0] = phi[0]
    if nst > 1:
        delta_phi[1:] = phi[1:] - phi[:-1]

    ref = delta_phi[0]
    if abs(ref) < 1e-10:
        # Degenerate mode (e.g. rigid body translation not present); fall back to uniform
        warnings.warn("First inter-storey mode-shape increment is near zero; using uniform drift ratios.")
        drift_ratios = np.ones(nst)
    else:
        drift_ratios = delta_phi / ref

    # Ensure drift ratios remain positive (negative drift implies mode reversal; clip at small positive)
    drift_ratios = np.maximum(drift_ratios, 1e-6)

    return shear_ratios, drift_ratios


def transform_sdof_to_mdof(Sd: np.ndarray,
                           Sa: np.ndarray,
                           phi: np.ndarray,
                           M: np.ndarray,
                           Gamma: float,
                           M_eff: float,
                           shear_ratios: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform SDOF capacity to MDOF storey force-drift relationships.

    Uniform-ductility assumption: all storeys share the same displacement
    backbone.  Every storey drift equals the first-storey drift (drift_ratio
    forced to 1.0), so only storey forces vary with height via shear_ratios[i].
    Only the first mode is used; no higher-mode correction is applied.

    Parameters
    ----------
    Sd : np.ndarray
        Spectral displacement (m).
    Sa : np.ndarray
        Spectral acceleration (g).
    phi : np.ndarray
        First mode shape (roof-normalised so phi[-1] = 1).
    M : np.ndarray
        Mass matrix.
    Gamma : float
        Modal participation factor.
    M_eff : float
        Effective modal mass.
    shear_ratios : np.ndarray
        Storey shear ratios V_i / V_base for each storey.

    Returns
    -------
    storey_forces : np.ndarray
        Forces in g × mass units (nst × n_points).
    storey_drifts : np.ndarray
        Interstorey drifts in m (nst × n_points).
    """
    nst = len(phi)
    n_points = len(Sd)

    # Base shear (g × mass)
    base_shear = Sa * M_eff

    # First inter-storey drift: δ_0 = Sd × Γ × φ[0]
    # Under uniform ductility every storey shares this same drift backbone.
    first_storey_drift = Sd * Gamma * phi[0]

    storey_forces = np.zeros((nst, n_points))
    storey_drifts = np.zeros((nst, n_points))

    for i in range(nst):
        storey_forces[i, :] = base_shear * shear_ratios[i]
        storey_drifts[i, :] = first_storey_drift          # same for all storeys

    return storey_forces, storey_drifts


def compute_floor_masses(phi: np.ndarray, M: np.ndarray) -> List[float]:
    """
    Return the diagonal floor masses from the mass matrix.

    The modeller assigns each floor node a mass equal to these values
    (in tonnes, or whatever consistent units are used). They sum to M_total.
    """
    return np.diag(M).tolist()


# =============================================================================
# PERIOD MATCHING
# =============================================================================

def correct_forces_for_period(storey_forces: np.ndarray,
                               T_actual: float,
                               T_target: float) -> Tuple[np.ndarray, float]:
    """
    Correct storey forces to achieve target period.

    Parameters
    ----------
    storey_forces : np.ndarray
        Current forces.
    T_actual : float
        Actual period from OpenSees.
    T_target : float
        Target period.

    Returns
    -------
    corrected_forces : np.ndarray
    correction_factor : float
    """
    correction = (T_actual / T_target) ** 2
    return storey_forces * correction, correction


# =============================================================================
# VALIDATION
# =============================================================================

def validate_capacity_curve(drifts: np.ndarray,
                            forces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure monotonic displacement and positive values."""
    drifts = np.abs(np.asarray(drifts, dtype=float))
    forces = np.abs(np.asarray(forces, dtype=float))

    for i in range(1, len(drifts)):
        if drifts[i] <= drifts[i - 1]:
            drifts[i] = drifts[i - 1] * 1.01

    return drifts, forces


# =============================================================================
# SYSTEM-AWARE AUTO-CONFIGURATION
# =============================================================================


def get_default_config(nst: int, isSOS: bool = False) -> CalibrationConfig:
    """Return a CalibrationConfig with fixed defaults for all systems."""
    return CalibrationConfig(
        soft_storey_factor = 0.35 if isSOS else 0.50,
        validate_results   = True,
    )


# =============================================================================
# MAIN CALIBRATION FUNCTION
# =============================================================================

def calibrate_model(
        nst: int,
        sdof_capacity: np.ndarray,
        isSOS: bool = False,
        storey_heights: Optional[List[float]] = None,
        stiffness_profile: Optional[np.ndarray] = None,
        mass_profile: Optional[np.ndarray] = None,
        phi_target: Optional[np.ndarray] = None,
        max_iterations: int = 500,
        config: Optional[CalibrationConfig] = None,
        verbose: bool = False) -> Tuple:
    """
    Calibrate MDOF storey force-deformation relationships from SDOF capacity.

    The target period is derived from the first capacity point:
        T_target = 2pi * sqrt(Sd[0] / (Sa[0] * g))

    Without storey_heights — analytical only, returns 5-tuple:
        floor_masses, storey_drifts (m), storey_forces (g x mass), phi, metadata

    With storey_heights — builds OpenSees model, iterates period to T_target,
    runs SPO for verification, returns same 5-tuple (no model returned; caller
    should reconstruct a clean model from storey_drifts and storey_forces).

    Parameters
    ----------
    nst : int
        Number of storeys.
    sdof_capacity : np.ndarray
        SDOF capacity [Sd (m), Sa (g)], shape (n, 2). Must not include (0, 0).
    isSOS : bool
        True if a soft storey is present. Reduces ground-floor stiffness by
        soft_storey_factor (0.35). Default False.
    storey_heights : list of float, optional
        Storey heights (m), length nst. Triggers OpenSees period iteration.
    stiffness_profile : np.ndarray, optional
        Custom relative storey stiffnesses (length nst). Overrides default.
    mass_profile : np.ndarray, optional
        Custom floor masses (length nst, must sum to 1.0). Overrides default.
    phi_target : np.ndarray, optional
        Custom first mode shape (length nst, roof-normalised). Overrides
        eigenvalue-based mode shape.
    max_iterations : int
        Maximum period-correction iterations. Default 500.
    config : CalibrationConfig, optional
        Advanced configuration. Auto-generated if omitted.
    verbose : bool
        Print iteration progress. Default False.
    """
    if config is None:
        config = get_default_config(nst, isSOS)
    config.verbose = verbose

    # ── Sanitise capacity curve ───────────────────────────────────────────────
    sdof_capacity = np.atleast_2d(np.asarray(sdof_capacity, dtype=float))
    nonzero = ~((sdof_capacity[:, 0] == 0) & (sdof_capacity[:, 1] == 0))
    if not np.all(nonzero):
        warnings.warn(f"Stripped {int(np.sum(~nonzero))} leading (0,0) row(s) from sdof_capacity.")
        sdof_capacity = sdof_capacity[nonzero]
    if len(sdof_capacity) < 2:
        raise ValueError("sdof_capacity must have at least 2 non-zero points.")
    Sd, Sa = sdof_capacity[:, 0], sdof_capacity[:, 1]
    if Sd[0] <= 0 or Sa[0] <= 0:
        raise ValueError(f"First capacity point must have Sd>0 and Sa>0. Got Sd={Sd[0]}, Sa={Sa[0]}.")

    # ── T_target from first capacity point ───────────────────────────────────
    T_target = 2 * np.pi * np.sqrt(Sd[0] / (Sa[0] * G))

    # ── Modal matrices ────────────────────────────────────────────────────────
    category = classify_building(nst)
    M = build_mass_matrix(nst, config.roof_mass_factor, mass_profile)
    K = build_stiffness_matrix(nst, isSOS,
                                soft_storey_factor=config.soft_storey_factor,
                                stiffness_profile=stiffness_profile,
                                group_size=config.stiffness_group_size)
    modal_props = compute_modal_properties(M, K, num_modes=1)

    # ── Mode shape ────────────────────────────────────────────────────────────
    if phi_target is not None:
        phi   = np.asarray(phi_target, dtype=float) / phi_target[-1]
        ones  = np.ones(nst)
        Gamma = float((phi @ M @ ones) / (phi @ M @ phi))
        M_eff = float((phi @ M @ ones) ** 2 / (phi @ M @ phi))
    else:
        phi   = modal_props['mode_shapes'][:, 0]
        Gamma = float(modal_props['participation_factors'][0])
        M_eff = float(modal_props['effective_masses'][0])

    MPR_mode1 = M_eff / np.trace(M)
    shear_ratios, drift_ratios = compute_storey_distribution_ratios(phi, M)

    # ── SDOF -> MDOF ──────────────────────────────────────────────────────────
    storey_forces, storey_drifts = transform_sdof_to_mdof(
        Sd, Sa, phi, M, Gamma, M_eff, shear_ratios)

    pc = 1.0   # no analytical pre-correction: T_target = T_implied by construction

    floor_masses = compute_floor_masses(phi, M)

    if config.validate_results:
        for i in range(nst):
            storey_drifts[i, :], storey_forces[i, :] = validate_capacity_curve(
                storey_drifts[i, :], storey_forces[i, :])

    # ── Metadata ──────────────────────────────────────────────────────────────
    metadata = {
        'building_category':             category.value,
        'num_storeys':                   nst,
        'T1':                            modal_props['periods'][0],
        'T_target':                      T_target,
        'period_correction':             pc,
        'Gamma':                         Gamma,
        'M_eff':                         M_eff,
        'MPR_mode1':                     MPR_mode1,
        'shear_ratios':                  shear_ratios,
        'drift_ratios':                  drift_ratios,
        'M':                             M,
        'K':                             K,
        'stiffness_group_size':          config.stiffness_group_size,
        'participation_factors':         modal_props['participation_factors'],
        'effective_masses':              modal_props['effective_masses'],
        'mass_participation_ratios':     modal_props['mass_participation_ratios'],
        'cumulative_mass_participation': modal_props['cumulative_mass_participation'],
    }

    if storey_heights is None:
        return floor_masses, storey_drifts, storey_forces, phi, metadata

    # =========================================================================
    # OpenSees period iteration + SPO
    # =========================================================================
    u_roof_target   = Sd * Gamma * phi[-1]
    u_roof_ultimate = u_roof_target[-1]

    if verbose:
        print("=" * 60)
        print(f"calibrate_model  nst={nst}  T_target={T_target:.4f} s")
        print(f"  Gamma={Gamma:.4f}  M_eff={M_eff:.4f}")
        print("=" * 60)

    T_opensees = None
    period_error = np.inf
    converged = False
    spo_Sd = spo_Sa = spo_roof_disp = spo_base_shear = None

    for iteration in range(max_iterations):
        if verbose:
            print("-" * 50 + f"  iter {iteration + 1}/{max_iterations}")
        try:
            model = _modeller(nst, storey_heights, floor_masses,
                              storey_drifts, storey_forces * G, False)
            model.compile_model()
            model.do_gravity_analysis()
            T_arr, _ = model.do_modal_analysis(num_modes=min(nst, 3), plot_modes=False)
            T_opensees   = T_arr[0]
            period_error = abs(T_opensees - T_target) / T_target
            if verbose:
                print(f"  [T]  T={T_opensees:.4f} s  target={T_target:.4f} s  err={period_error:.2%}")
        except Exception as e:
            warnings.warn(f"Model build failed at iteration {iteration + 1}: {e}")
            break

        if period_error <= config.period_tolerance:
            converged = True
            if verbose:
                print(f"Converged  T={T_opensees:.4f} s  pc={pc:.4f}x")
            break

        c = float(np.clip(1.0 + 0.8 * ((T_opensees / T_target) ** 2 - 1.0), 0.5, 3.0))
        storey_forces = storey_forces * c
        pc *= c
        if verbose:
            print(f"  [T]  Correction {c:.4f}x forces  (cumulative pc={pc:.4f})")
    else:
        if verbose:
            print(f"Did not converge  period error={period_error:.2%}")

    # ── SPO for verification ──────────────────────────────────────────────────
    try:
        model = _modeller(nst, storey_heights, floor_masses,
                          storey_drifts, storey_forces * G, False)
        model.compile_model()
        model.do_gravity_analysis()
        model.do_modal_analysis(num_modes=min(nst, 3), plot_modes=False)
        ref_disp   = max(u_roof_target[0], 1e-4)
        disp_scale = max((u_roof_ultimate * 1.5) / ref_disp, 2.0)
        spo_res    = model.do_spo_analysis(
            ref_disp=ref_disp, disp_scale_factor=disp_scale,
            push_dir=1, phi=phi.tolist(), pflag=False)
        spo_roof_disp  = spo_res['spo_disps'][:, -1]
        spo_base_shear = spo_res['spo_rxn']
        spo_Sd         = spo_roof_disp / (Gamma * phi[-1])
        spo_Sa         = spo_base_shear / (G * M_eff)
        if verbose:
            for k_pt, sd_t in enumerate(Sd):
                sa_norm = float(np.interp(sd_t, spo_Sd, spo_Sa)) / pc
                print(f"  [SPO] Pt{k_pt+1}: Sd={sd_t:.4f}  Sa_spo/pc={sa_norm:.4f}g  Sa_target={Sa[k_pt]:.4f}g")
    except Exception as e:
        warnings.warn(f"SPO failed: {e}")

    metadata.update({
        'T_achieved':              T_opensees,
        'period_error_final':      period_error,
        'converged':               converged,
        'spo_iterations':          iteration + 1,
        'period_correction':       pc,
        'u_roof_target':           u_roof_target,
    })
    if spo_Sd is not None:
        metadata.update({
            'spo_sdof_Sd':             spo_Sd,
            'spo_sdof_Sa':             spo_Sa,
            'spo_sdof_Sa_normalised':  spo_Sa / pc,
            'spo_roof_disp':           spo_roof_disp,
            'spo_base_shear':          spo_base_shear,
        })

    return floor_masses, storey_drifts, storey_forces, phi, metadata


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def mdof_to_sdof(base_shear: np.ndarray,
                 roof_disp: np.ndarray,
                 Gamma: float,
                 M_eff: float,
                 phi_roof: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert MDOF pushover results to SDOF coordinates.

    Parameters
    ----------
    base_shear : np.ndarray
        Base shear (kN).
    roof_disp : np.ndarray
        Roof displacement (m).
    Gamma : float
        Modal participation factor.
    M_eff : float
        Effective modal mass.
    phi_roof : float
        Roof mode shape value (usually 1.0).

    Returns
    -------
    Sd : np.ndarray
        Spectral displacement (m).
    Sa : np.ndarray
        Spectral acceleration (g).
    """
    Sd = roof_disp / (Gamma * phi_roof)
    Sa = base_shear / (G * M_eff)
    return Sd, Sa


def sdof_to_mdof_global(Sd: np.ndarray,
                        Sa: np.ndarray,
                        Gamma: float,
                        M_eff: float,
                        phi_roof: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert SDOF capacity to MDOF global response.

    Returns
    -------
    V_base : np.ndarray
        Base shear (kN).
    u_roof : np.ndarray
        Roof displacement (m).
    """
    V_base = Sa * G * M_eff
    u_roof = Sd * Gamma * phi_roof
    return V_base, u_roof
