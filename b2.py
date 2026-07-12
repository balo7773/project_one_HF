"""
Airfoil Bayesian Optimizer
===========================
Searches the CST parameter space to find the airfoil geometry
that maximises CL/CD at user-specified flight conditions,
subject to a pitching moment stability constraint (CM > threshold).

After finding the best airfoil, automatically runs a SHAP explanation
on that single result — telling the user WHY that geometry was chosen
in plain aerodynamic terms.

Install dependencies first (once):
    pip install scikit-optimize --break-system-packages
    pip install shap --break-system-packages

Run:
    python bayesian_optimizer.py

Or with command-line arguments:
    python bayesian_optimizer.py --re 500000 --alpha 5.0 --n_calls 150

Stability mode:
    python bayesian_optimizer.py --stability A   # tailless / flying wing
    python bayesian_optimizer.py --stability B   # conventional (default)
    python bayesian_optimizer.py --stability C   # large tail / relaxed
    python bayesian_optimizer.py --stability D   # no constraint

Outputs saved to AIRFOIL_FILES/optimizer_results/:
    best_airfoil.dat           ← coordinates ready for XFOIL validation
    best_airfoil.png           ← airfoil shape plot
    convergence.png            ← objective vs iteration
    shap_explanation.png       ← why this airfoil was recommended
    optimization_report.txt    ← full summary of what was found
    all_evaluations.csv        ← every CST combo tried + its CL/CD/CM
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
import streamlit as st
from huggingface_hub import hf_hub_download # new for cloud eployment
import streamlit as st # new for cloud deployment
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import comb
import warnings
warnings.filterwarnings('ignore')

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
except ImportError:
    print('ERROR: PyTorch not found. Activate your conda environment first.')
    sys.exit(1)

try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.plots import plot_convergence
except ImportError:
    print('ERROR: scikit-optimize not found.')
    print('Install with: pip install scikit-optimize --break-system-packages')
    sys.exit(1)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print('⚠  shap not found — SHAP explanation will be skipped.')
    print('   Install with: pip install shap --break-system-packages')
    SHAP_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

#BASE_DIR     = os.path.dirname(os.path.abspath(__file__))  # always relative to this script
#MODEL_PATH   = os.path.join(BASE_DIR, 'model', 'best_model.pt')
#SCALER_PATH  = os.path.join(BASE_DIR, 'scaler_params.json')
#DATA_CSV     = os.path.join(BASE_DIR, 'master_dataset_clean.csv')  # clean dataset — corrupt CST fits removed
# ── Output directory ─────────────────────────────────────────────────────────
# RESULTS_ROOT is the parent folder — always exists.
# OUT_DIR is set at runtime to a per-run timestamped subfolder:
#   optimizer_results/
#     run_20250307_143022_single_Re500000_a8.0/
#     run_20250307_151204_multi_phase/
#     latest -> symlink updated each run (for dashboard)
#
# OUT_DIR is re-assigned in _setup_run_dir() before any file writes.
# Module-level assignment here is just a safe default — never written to.
BBASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(BASE_DIR, 'optimizer_results')
os.makedirs(RESULTS_ROOT, exist_ok=True)
OUT_DIR = RESULTS_ROOT
# Define your exact Hugging Face repository
HF_REPO_ID = "balo7773/project_one" 

def load_hf_assets():
    """Fetches ONLY the heavy model and scaler from Hugging Face."""
    try:
        model = hf_hub_download(repo_id=HF_REPO_ID, filename="best_model.pt")
        scaler = hf_hub_download(repo_id=HF_REPO_ID, filename="scaler_params.json")
        return model, scaler
    except Exception as e:
        print(f"Failed to load assets from Hugging Face: {e}")
        sys.exit(1)

# Retrieve cloud assets
MODEL_PATH, SCALER_PATH = load_hf_assets()

# Point directly to the local dataset cloned by Streamlit
DATA_CSV = os.path.join(BASE_DIR, 'master_dataset_clean.csv')

def _setup_run_dir(tag: str) -> str:
    """
    Create a timestamped subfolder for this run and update the 'latest'
    symlink so the dashboard always finds the most recent result.

    tag   — short descriptor appended to folder name, e.g.
            'single_Re500000_a8.0' or 'multi_phase'

    Returns the absolute path to the new run directory.
    """
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder    = f'run_{timestamp}_{tag}'
    run_dir   = os.path.join(RESULTS_ROOT, folder)
    os.makedirs(run_dir, exist_ok=True)

    # Update 'latest' pointer — used by dashboard to find current run
    latest_path = os.path.join(RESULTS_ROOT, 'latest')
    try:
        if os.path.islink(latest_path) or os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(run_dir, latest_path)
    except OSError:
        # Windows fallback — write a text file with the path
        try:
            with open(latest_path + '.txt', 'w') as f:
                f.write(run_dir)
        except Exception:
            pass

    return run_dir

# ── Default flight conditions (override via command line) ─────────────────────
DEFAULT_RE    = 500_000   # Reynolds number
DEFAULT_ALPHA = 5.0       # angle of attack in degrees
DEFAULT_NCALLS= 150       # number of Bayesian optimization iterations

# ── Optimization mode ────────────────────────────────────────────────────────
# Mode 1 (target_lift): user specifies CL target → minimise CD
# Mode 2 (best_efficiency): free optimisation → maximise CL/CD
DEFAULT_MODE      = 'best_efficiency'
CL_TOLERANCE      = 0.05   # ±0.05 around target — user never sees this
# Maps a user-friendly label → (CM_MIN threshold, description)
# The optimizer uses CM_MIN internally — user never sees raw numbers.
STABILITY_MODES = {
    'A': (-0.01, 'Tailless / flying wing   — very strict  (CM > -0.01)'),
    'B': (-0.05, 'Conventional aircraft    — standard     (CM > -0.05)'),
    'C': (-0.10, 'Aircraft with large tail — relaxed      (CM > -0.10)'),
    'D': (-9999, 'No constraint            — pure CL/CD maximisation'),
}
DEFAULT_STABILITY = 'B'

CM_PENALTY       = 10.0   # penalty weight when CM constraint violated
                           # higher = stricter enforcement of stability



# ── CST reconstruction ────────────────────────────────────────────────────────
N_CST         = 10       # params per surface (must match training)
N_RECONSTRUCT = 200      # coordinate points for output airfoil shape

# ══════════════════════════════════════════════════════════════════════════════
#  MLP ARCHITECTURE — must exactly match train_mlp.py
# ══════════════════════════════════════════════════════════════════════════════

class AirfoilMLP(nn.Module):
    """
    Shared trunk -> separate heads for CL, CD, CM.

    ADJUSTMENT 1: CD head receives trunk output (256) + Re (1) + alpha (1) = 258-dim input.
    Re and alpha bypass the trunk and arrive fresh at the CD head,
    giving it a sharp, unblurred flight-condition signal for drag prediction.

    Input feature order:
        indices 0-19  : CST coefficients (20 features)
        index   20    : xtr_top
        index   21    : xtr_bot
        index   22    : re_normalized       <- routed directly to CD head
        index   23    : alpha_normalized    <- routed directly to CD head
    """
    def __init__(self, input_dim, hidden_layers, activation, dropout_rate):
        super().__init__()
        act_map = {
            'relu': nn.ReLU, 'gelu': nn.GELU,
            'tanh': nn.Tanh, 'silu': nn.SiLU,
        }
        act_cls = act_map.get(activation, nn.GELU)

        trunk_layers = []
        in_dim = input_dim
        for h_dim in hidden_layers[:-1]:
            trunk_layers += [
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                act_cls(),
                nn.Dropout(dropout_rate),
            ]
            in_dim = h_dim
        self.trunk = nn.Sequential(*trunk_layers)

        head_dim = hidden_layers[-1]

        # CL and CM heads: unchanged, receive trunk output (256-dim)
        self.cl_head = nn.Sequential(
            nn.Linear(in_dim, head_dim), act_cls(),
            nn.Linear(head_dim, 1)
        )
        self.cm_head = nn.Sequential(
            nn.Linear(in_dim, head_dim), act_cls(),
            nn.Linear(head_dim, 1)
        )

        # ADJUSTMENT 1: CD head receives trunk(256) + Re(1) + alpha(1) = 258-dim
        cd_head_input_dim = in_dim + 2
        self.cd_head = nn.Sequential(
            nn.Linear(cd_head_input_dim, head_dim), act_cls(),
            nn.Linear(head_dim, 1)
        )

        # Indices of Re and alpha in the input vector
        self._re_idx    = input_dim - 2  # index 22
        self._alpha_idx = input_dim - 1  # index 23

    def forward(self, x):
        f = self.trunk(x)
        # ADJUSTMENT 1: concatenate Re and alpha directly to trunk output for CD head
        re_alpha = x[:, self._re_idx:self._alpha_idx + 1]  # shape (batch, 2)
        cd_input = torch.cat([f, re_alpha], dim=1)          # shape (batch, 258)
        return torch.cat([self.cl_head(f),
                          self.cd_head(cd_input),
                          self.cm_head(f)], dim=1)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path, device):
    print(f'  Loading model from {model_path}')
    ckpt = torch.load(model_path, map_location=device)
    cfg  = ckpt['cfg']

    model = AirfoilMLP(
        input_dim    = 22,  # 20 CST + Re + alpha
        hidden_layers= cfg['hidden_layers'],
        activation   = cfg['activation'],
        dropout_rate = 0.0,   # CRITICAL: dropout OFF at inference
    ).to(device)

    model.load_state_dict(ckpt['model_state'])
    model.eval()

    print(f'  Loaded — best epoch {ckpt["epoch"]}  '
          f'val R² CL={ckpt["val_r2"][0]:.4f}  '
          f'CD={ckpt["val_r2"][1]:.4f}  '
          f'CM={ckpt["val_r2"][2]:.4f}')
    return model, cfg


def load_scalers(path):
    with open(path, 'r') as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE — normalize inputs, forward pass, inverse transform outputs
# ══════════════════════════════════════════════════════════════════════════════

def predict(model, scalers, cst_params, re_val, alpha_val, device):
    """
    Given 20 CST parameters + Re + alpha → return physical CL, CD, CM.

    This function handles the full normalization/denormalization cycle
    so the optimizer never has to think about scaled values.

    Steps:
      1. Normalize all 24 inputs using scaler_params.json
      2. Forward pass through MLP
      3. Inverse transform outputs back to physical units
         (CD specifically: reverse StandardScaler then exp())
    """
    cst_cols = [f'cst_u{i}' for i in range(10)] + \
               [f'cst_l{i}' for i in range(10)]

    # ── Normalize inputs ──────────────────────────────────────────────────────
    x = np.zeros(22, dtype=np.float32)
    for i, col in enumerate(cst_cols):
        s = scalers[col]
        x[i] = (cst_params[i] - s['mean']) / (s['std'] + 1e-10)

    s = scalers['re']
    x[20] = (re_val - s['mean']) / (s['std'] + 1e-10)

    s = scalers['alpha']
    x[21] = (alpha_val - s['mean']) / (s['std'] + 1e-10)

    # ── Forward pass ──────────────────────────────────────────────────────────
    with torch.no_grad():
        tensor_in  = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
        tensor_out = model(tensor_in).cpu().numpy()[0]

    # ── Inverse transform outputs ─────────────────────────────────────────────
    s_cl = scalers['cl']
    cl   = tensor_out[0] * s_cl['std'] + s_cl['mean']

    s_cd = scalers['cd']
    cd   = np.exp(tensor_out[1] * s_cd['log_std'] + s_cd['log_mean'])

    s_cm = scalers['cm']
    cm   = tensor_out[2] * s_cm['std'] + s_cm['mean']

    return float(cl), float(cd), float(cm)


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH BOUNDS — derived from actual training data
# ══════════════════════════════════════════════════════════════════════════════

def compute_cst_bounds(data_csv, margin=0.15):
    """
    Compute search bounds for each CST parameter from training data min/max
    with a ±15% margin extension.

    Why min/max ±15% (not percentiles):
    p5/p95 sounds conservative but is actually too restrictive in high
    dimensions. With 20 CST parameters, cutting each dimension to 42% of
    its original range leaves only 42^20 ≈ 3e-8 of the original search
    volume. The optimizer cannot reach legitimate high-performance geometries
    (Selig S1223, Eppler 423) that sit near the edges of the training
    distribution. It compensates by creating wavy multi-humped surfaces —
    geometries that satisfy the tight bounds through unusual coefficient
    combinations rather than through good aerodynamic design.

    The geometry validity check (_geometry_is_valid) is the correct guard
    against physically impossible geometries. Bounds should be generous
    enough to include all real high-performance airfoils, with the validity
    check catching anything that becomes physically self-intersecting.

    The ±15% margin gives the optimizer a small exploration zone just beyond
    the most extreme training examples — useful for discovering new shapes
    similar to but slightly beyond the training boundary.

    Returns list of (lo, hi) tuples for each of the 20 CST columns.
    """
    print('  Computing CST bounds from training data...')
    df = pd.read_csv(data_csv)
    train_df = df[df['split'] == 'train']

    cst_cols = [f'cst_u{i}' for i in range(10)] + \
               [f'cst_l{i}' for i in range(10)]
    bounds = []
    for col in cst_cols:
        col_data = train_df[col]
        lo   = col_data.min()
        hi   = col_data.max()
        span = hi - lo
        lo_m = lo - margin * span
        hi_m = hi + margin * span
        bounds.append((float(lo_m), float(hi_m)))

    print(f'  Bounds computed for {len(bounds)} CST parameters (min/max ±{margin*100:.0f}% margin)')
    return bounds, cst_cols


# ══════════════════════════════════════════════════════════════════════════════
#  OBJECTIVE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

# Global state for the objective function (skopt requires a single callable)
_model      = None
_scalers    = None
_device     = None
_re         = None
_alpha      = None
_cm_min     = None   # set from stability mode selection
_cl_target  = None   # set when mode = target_lift, else None
_opt_mode   = None   # 'target_lift' or 'best_efficiency'
_eval_log   = []     # stores every evaluation for analysis


# ── Geometry validity check ──────────────────────────────────────────────────
# Pre-filter applied BEFORE every MLP call.
# Reconstructs the airfoil at 50 chord stations and checks two conditions:
#
#   1. y_upper(x) > y_lower(x) at every station  — no surface crossing
#   2. thickness = y_upper - y_lower > MIN_THICKNESS everywhere — no degenerate sections
#
# If either fails, the geometry is physically impossible.
# Return 999.0 immediately and skip the MLP entirely.
#
# Cost: ~0.1ms per call (pure numpy, no model inference).
# The MLP call costs ~5ms. This filter costs 2% of that — worth it every time.
#
# Analogy: a cook smells the milk before adding it to the batter.
# Takes one second. Saves the whole dish.

# Two separate x-grids for geometry checks:
#   _CHECK_X_FULL  — full chord, used only for surface crossing check
#   _CHECK_X_MID   — mid-chord only (5%–95%), used for thickness checks
#
# Why separate: real airfoils naturally taper to near-zero thickness at
# both the leading and trailing edges. Checking MIN_THICKNESS at x=0.01
# or x=0.99 rejects every real sharp-edged airfoil. The knife-edge check
# only makes physical sense in the middle of the chord where the section
# should have meaningful structural depth.
_CHECK_X_FULL = np.linspace(0.01, 0.99, 50)   # surface crossing check
_CHECK_X_MID  = np.linspace(0.10, 0.65, 30)   # TE taper is expected — only check forward 2/3   # thickness checks (mid-chord only)
MIN_THICKNESS = 0.002   # 0.2% chord — forward-section knife-edge threshold (TE taper expected past x=0.65)
MAX_THICKNESS = 0.400   # 40.0% chord — balloon shape ceiling


def _geometry_is_valid(cst_params):
    """
    Returns (True, None) if geometry is physically valid.
    Returns (False, reason_string) if not.
    Fast — no MLP involved. Runs BEFORE every MLP call.

    Three checks in order of severity:

    1. No surface crossing  — upper must always be above lower
    2. No knife-edge section — min thickness > 0.5% chord
    3. No balloon shape     — max thickness < 21% chord

    The third check is what prevents the optimizer from exploiting
    the wide min/max bounds by proposing giant unrealistic geometries.
    A real airfoil is never taller than ~21% of its own length.
    The training data MLP has never seen anything like that — its
    predictions there are pure hallucination.
    """
    wu = np.array(cst_params[:10])
    wl = np.array(cst_params[10:20])

    # Check 1 — surface crossing (full chord)
    yu_full = reconstruct_surface(_CHECK_X_FULL, wu)
    yl_full = reconstruct_surface(_CHECK_X_FULL, wl)
    t_full  = yu_full - yl_full
    if np.any(t_full <= 0):
        n_bad = int(np.sum(t_full <= 0))
        worst = float(t_full.min())
        return False, f'surface crossing: {n_bad} stations, min={worst:.4f}'

    # Check 2 & 3 — knife-edge and balloon (mid-chord only, 5%–95%)
    # Trailing and leading edge naturally thin to near-zero — only check the
    # middle section where structural depth must be non-trivial.
    yu_mid = reconstruct_surface(_CHECK_X_MID, wu)
    yl_mid = reconstruct_surface(_CHECK_X_MID, wl)
    thickness = yu_mid - yl_mid

    # Check 2 — knife-edge / degenerate section
    if np.any(thickness < MIN_THICKNESS):
        n_thin = int(np.sum(thickness < MIN_THICKNESS))
        return False, f'knife-edge section: {n_thin} mid-chord stations below {MIN_THICKNESS*100:.1f}%'

    # Check 3 — oscillatory / Runge-phenomenon shape
    # Rule: upper surface must have exactly 1 peak (smooth rise to max
    # thickness, then taper). Lower surface must have exactly 1 trough
    # (smooth concave belly). These are true for 548/549 clean airfoils.
    # Allowing 2 peaks permits a "camel hump" — still wrong geometry.
    # n_peaks > 1 strictly enforces the single-hump constraint.
    x_smooth = np.linspace(0.01, 0.99, 100)
    yu_s = reconstruct_surface(x_smooth, wu)
    yl_s = reconstruct_surface(x_smooth, wl)
    dy_u = np.diff(yu_s)
    dy_l = np.diff(yl_s)

    # Upper surface: count peaks (slope goes + then -)
    n_upper_peaks = int(np.sum((dy_u[:-1] > 0) & (dy_u[1:] < 0)))
    if n_upper_peaks > 1:
        return False, f'oscillatory upper surface: {n_upper_peaks} peaks (must be 1)'

    # Lower surface: count troughs (slope goes - then +)
    n_lower_troughs = int(np.sum((dy_l[:-1] < 0) & (dy_l[1:] > 0)))
    if n_lower_troughs > 1:
        return False, f'oscillatory lower surface: {n_lower_troughs} troughs (must be ≤1)'

    # Check 4 — balloon / physically impossible thickness
    max_t = float(thickness.max())
    if max_t > MAX_THICKNESS:
        return False, f'airfoil too thick: max thickness={max_t*100:.1f}% chord (limit {MAX_THICKNESS*100:.0f}%)'

    # Check 5 — Leading edge "cliff drop" (Gemini check A)
    # Real subsonic lower surfaces enter the flow gently near the LE.
    # A slope steeper than -0.90 at x=0.01→0.05 means an immediate
    # violent plunge — causes flow separation, not seen in clean data.
    # 541/549 clean airfoils pass this threshold.
    x_le   = np.linspace(0.01, 0.05, 20)
    yl_le  = reconstruct_surface(x_le, wl)
    le_slope = (yl_le[-1] - yl_le[0]) / (x_le[-1] - x_le[0])
    if le_slope < -0.90:
        return False, f'LE cliff drop: lower surface slope={le_slope:.3f} < -0.90'

    # Check 6 — Max thickness location (Gemini check B)
    # Real airfoils reach max thickness between x=0.10 and x=0.50.
    # A shape with max thickness at x=0.03 or x=0.80 is not aerodynamic.
    # 541/549 clean airfoils pass this window.
    x_chk    = np.linspace(0.01, 0.99, 200)
    yu_chk   = reconstruct_surface(x_chk, wu)
    yl_chk   = reconstruct_surface(x_chk, wl)
    t_chk    = yu_chk - yl_chk
    tmax_xi  = x_chk[np.argmax(t_chk)]
    if not (0.10 <= tmax_xi <= 0.50):
        return False, f'max thickness at x={tmax_xi:.3f} outside [0.10, 0.50]'

    return True, None


def _get_thickness_stats(cst_params):
    """Return (min_thickness, max_thickness) as % chord. Used for reporting."""
    wu = np.array(cst_params[:10])
    wl = np.array(cst_params[10:20])
    yu = reconstruct_surface(_CHECK_X_FULL, wu)
    yl = reconstruct_surface(_CHECK_X_FULL, wl)
    thickness = yu - yl
    return float(thickness.min()) * 100, float(thickness.max()) * 100


def objective(cst_params):
    """
    Objective function supporting two modes:

    MODE 1 — target_lift:
        User wants a specific CL. Job is to find the geometry that
        hits that CL with the lowest possible CD.
        Objective  = CD  (minimise directly)
        Constraint = |CL - CL_target| < CL_TOLERANCE  (soft penalty)
        Constraint = CM > _cm_min                      (soft penalty)

        Analogy: you need to carry exactly 10kg. Find the cheapest
        container that holds exactly 10kg — not 8, not 12. Exactly 10.

    MODE 2 — best_efficiency:
        No CL target. Find the geometry with the best lift-to-drag ratio.
        Objective  = -(CL/CD)  (minimise negative = maximise ratio)
        Constraint = CM > _cm_min  (soft penalty)

    Both modes use SOFT penalties — smooth gradients, no discontinuous
    cliffs — so the Gaussian Process can navigate the landscape cleanly.
    """
    global _model, _scalers, _device, _re, _alpha
    global _cm_min, _cl_target, _opt_mode, _eval_log

    # ── Geometry validity — runs BEFORE the MLP ───────────────────────────────
    # If the surface crosses itself or has degenerate thickness, the MLP
    # has no valid training examples near this geometry and will hallucinate.
    # Reject immediately — no inference needed.
    valid, reason = _geometry_is_valid(cst_params)
    if not valid:
        _eval_log.append({
            'cl': np.nan, 'cd': np.nan, 'cm': np.nan,
            'cl_cd': np.nan, 'objective': 999.0, 'feasible': False,
            'invalid_geometry': reason,
        })
        return 999.0

    cl, cd, cm = predict(_model, _scalers, cst_params, _re, _alpha, _device)

    # Guard against physically impossible predictions
    if cd <= 0 or np.isnan(cl) or np.isnan(cd) or np.isnan(cm):
        _eval_log.append({
            'cl': np.nan, 'cd': np.nan, 'cm': np.nan,
            'cl_cd': np.nan, 'objective': 999.0, 'feasible': False
        })
        return 999.0

    cl_cd = cl / cd

    # ── CM stability penalty (same in both modes) ─────────────────────────────
    cm_violation = max(0.0, _cm_min - cm)
    penalty      = CM_PENALTY * (cm_violation ** 2)

    # ── Mode-specific objective and CL penalty ────────────────────────────────
    # Guard: if globals somehow not set, default to best_efficiency so we never
    # silently mark a CL-violating result as feasible.
    current_mode   = _opt_mode  if _opt_mode  is not None else 'best_efficiency'
    current_target = _cl_target if _cl_target is not None else 0.0

    if current_mode == 'target_lift':
        # Penalise deviation from CL target — quadratic hard wall
        cl_violation = abs(cl - current_target) - CL_TOLERANCE
        cl_violation = max(0.0, cl_violation)
        # x20: at |dCL|=0.5, penalty=50 >> any realistic CD. A geometry
        # missing the lift target can NEVER win against one that hits it.
        penalty     += CM_PENALTY * (cl_violation ** 2) * 20

        objective_val = cd + penalty   # minimise drag directly

        # Feasible = STRICTLY within CL tolerance AND CM satisfied.
        # Computed from current_target — never inherited from a stale global.
        feasible = (abs(cl - current_target) <= CL_TOLERANCE
                    and cm >= _cm_min)

    else:  # best_efficiency
        if cl < 0:
            penalty += CM_PENALTY * abs(cl)
        objective_val = -cl_cd + penalty   # maximise CL/CD
        feasible      = (cm >= _cm_min and cl > 0)

    _eval_log.append({
        'cl'        : cl,
        'cd'        : cd,
        'cm'        : cm,
        'cl_cd'     : cl_cd,
        'objective' : objective_val,
        'feasible'  : feasible,
        'cst_params': list(cst_params),
    })

    return objective_val


# ══════════════════════════════════════════════════════════════════════════════
#  CST RECONSTRUCTION — turn weights back into airfoil coordinates
# ══════════════════════════════════════════════════════════════════════════════

def bernstein(x, n, k):
    return comb(n, k, exact=False) * (x**k) * ((1-x)**(n-k))

def class_fn(x):
    return (x ** 0.5) * ((1 - x) ** 1.0)

def reconstruct_surface(x, weights, te_y=0.0):
    """Reconstruct one surface (upper or lower) from CST weights."""
    xs    = np.clip(x, 1e-7, 1.0 - 1e-7)
    n     = len(weights) - 1
    C     = class_fn(xs)
    shape = sum(weights[k] * C * bernstein(xs, n, k) for k in range(len(weights)))
    return shape + te_y * x

def cst_to_coordinates(cst_params, n_points=N_RECONSTRUCT):
    """
    Convert 20 CST weights back to (x, y) airfoil coordinates.

    Upper surface: cst_params[0:10]
    Lower surface: cst_params[10:20]

    The output is in Selig format (TE → upper → LE → lower → TE)
    ready to be saved as a .dat file for XFOIL.
    """
    w_upper = np.array(cst_params[:10])
    w_lower = np.array(cst_params[10:])

    # Remove zero-padding — find last non-zero weight
    n_u = max(np.where(w_upper != 0)[0]) + 1 if np.any(w_upper != 0) else 4
    n_l = max(np.where(w_lower != 0)[0]) + 1 if np.any(w_lower != 0) else 4
    w_upper = w_upper[:n_u]
    w_lower = w_lower[:n_l]

    # Cosine-spaced x for smooth representation
    x = (1 - np.cos(np.linspace(0, np.pi, n_points))) / 2
    x = np.clip(x, 1e-7, 1.0 - 1e-7)

    y_upper = reconstruct_surface(x, w_upper)
    y_lower = reconstruct_surface(x, w_lower)

    # Selig format: TE → upper(reversed) → LE → lower
    upper_pts = np.column_stack([x[::-1], y_upper[::-1]])
    lower_pts = np.column_stack([x[1:],   y_lower[1:]])
    coords    = np.vstack([upper_pts, lower_pts])

    return coords, x, y_upper, y_lower


def save_dat_file(coords, name, path):
    """Save airfoil coordinates as a XFOIL-compatible .dat file."""
    with open(path, 'w') as f:
        f.write(f'{name}\n')
        for xc, yc in coords:
            f.write(f'  {xc:.7f}  {yc:.7f}\n')
    print(f'  Saved .dat → {path}')


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_convergence_curve(eval_log, re, alpha, save_path):
    global _opt_mode, _cl_target
    
    feasible = [e for e in eval_log if e['feasible']]
    if not feasible:
        print('  ⚠ No feasible solutions found for convergence plot')
        return

    best_so_far = []
    
    # Determine what we are tracking based on the mode
    if _opt_mode == 'target_lift':
        current_best = np.inf
        for e in eval_log:
            if e['feasible'] and not np.isnan(e['cd']):
                current_best = min(current_best, e['cd'])
            best_so_far.append(current_best if current_best < np.inf else np.nan)
        y_label = 'Best CD found (lower is better)'
        title_left = 'Convergence — Minimum CD vs Iteration'
    else:
        current_best = -np.inf
        for e in eval_log:
            if e['feasible'] and not np.isnan(e['cl_cd']):
                current_best = max(current_best, e['cl_cd'])
            best_so_far.append(current_best if current_best > -np.inf else np.nan)
        y_label = 'Best CL/CD ratio'
        title_left = 'Convergence — Best CL/CD vs Iteration'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Bayesian Optimization — Re={re:.0e}  α={alpha}°', fontsize=12, fontweight='bold')

    # Left Plot
    ax = axes[0]
    ax.plot(range(1, len(best_so_far)+1), best_so_far, 'steelblue', lw=2, label=y_label)
    ax.fill_between(range(1, len(best_so_far)+1), best_so_far, alpha=0.15, color='steelblue')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(y_label)
    ax.set_title(title_left)
    ax.legend(); ax.grid(True, alpha=0.3)

    # Right Plot
    ax = axes[1]
    all_cl = [e['cl'] for e in eval_log if not np.isnan(e.get('cl', np.nan))]
    all_cd = [e['cd'] for e in eval_log if not np.isnan(e.get('cd', np.nan))]
    feas   = [e['feasible'] for e in eval_log if not np.isnan(e.get('cl', np.nan))]

    if all_cl and all_cd:
        cl_arr = np.array(all_cl)
        cd_arr = np.array(all_cd)
        feas_arr = np.array(feas)
        ax.scatter(cd_arr[~feas_arr], cl_arr[~feas_arr], s=8, alpha=0.4, color='coral', label='Infeasible')
        ax.scatter(cd_arr[feas_arr],  cl_arr[feas_arr], s=8, alpha=0.6, color='steelblue', label='Feasible')

        # Mark best point correctly based on mode
        best_e = [e for e in eval_log if e['feasible']]
        if _opt_mode == 'target_lift':
            best_idx = np.argmin([e['cd'] for e in best_e])
        else:
            best_idx = np.argmax([e['cl_cd'] for e in best_e])
            
        b = best_e[best_idx]
        ax.scatter(b['cd'], b['cl'], s=120, color='gold', edgecolors='black', lw=1.5, zorder=5, label='Best')

        # If target_lift, draw a horizontal line showing the target CL
        if _opt_mode == 'target_lift' and _cl_target is not None:
            ax.axhline(_cl_target, color='green', linestyle='--', alpha=0.6, label=f'Target CL ({_cl_target})')

    ax.set_xlabel('CD'); ax.set_ylabel('CL')
    ax.set_title('All Evaluations — CL vs CD')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

def plot_airfoil(x, y_upper, y_lower, cl, cd, cm, re, alpha, save_path):
    """
    Plot the optimized airfoil shape with key aerodynamic numbers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Optimized Airfoil — Re={re:.0e}  α={alpha}°  '
        f'CL={cl:.3f}  CD={cd:.5f}  CM={cm:.4f}  CL/CD={cl/cd:.1f}',
        fontsize=11, fontweight='bold'
    )

    # Full airfoil shape
    ax = axes[0]
    ax.plot(x[::-1], y_upper[::-1], 'steelblue', lw=2, label='Upper surface')
    ax.plot(x,       y_lower,       'coral',     lw=2, label='Lower surface')
    ax.fill_between(
        np.concatenate([x[::-1], x[1:]]),
        np.concatenate([y_upper[::-1], y_lower[1:]]),
        alpha=0.08, color='steelblue'
    )
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
    ax.set_aspect('equal')
    ax.set_xlabel('x/c'); ax.set_ylabel('y/c')
    ax.set_title('Optimized Airfoil Shape')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Leading edge zoom
    ax = axes[1]
    mask = x <= 0.25
    ax.plot(x[::-1][x[::-1] <= 0.25], y_upper[::-1][x[::-1] <= 0.25],
            'steelblue', lw=2.5)
    ax.plot(x[mask], y_lower[mask], 'coral', lw=2.5)
    ax.set_aspect('equal')
    ax.set_xlabel('x/c'); ax.set_ylabel('y/c')
    ax.set_title('Leading Edge Detail (x < 0.25)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Airfoil plot → {save_path}')

def plot_airfoil_multiphase_stacked(x, y_upper, y_lower, phase_results, phases, save_path):
    """
    Creates a vertically stacked image of the standard airfoil plot, 
    one row for each flight phase evaluated in the multi-phase run.
    """
    n_phases = len(phase_results)
    # Height scales dynamically: 5 inches per phase
    fig, axes = plt.subplots(n_phases, 2, figsize=(14, 5 * n_phases))
    fig.suptitle('Multi-Phase Optimized Airfoil — Evaluated Across All Phases', 
                 fontsize=14, fontweight='bold', y=0.98)

    for i, (pr, ph) in enumerate(zip(phase_results, phases)):
        # If there's only 1 phase, axes is 1D. If multiple, it's 2D.
        ax_full = axes[i, 0] if n_phases > 1 else axes[0]
        ax_le   = axes[i, 1] if n_phases > 1 else axes[1]

        cl = pr['cl']
        cd = pr['cd']
        cm = pr['cm']
        ld = cl / cd if cd > 0 else 0
        name = str(pr['name']).capitalize()
        re = ph['re']
        alpha = ph['alpha']

        title_str = f'[{name}] Re={re:.0e}  α={alpha}°  CL={cl:.3f}  CD={cd:.5f}  CM={cm:.4f}  CL/CD={ld:.1f}'

        # --- LEFT: Full Airfoil ---
        ax_full.plot(x[::-1], y_upper[::-1], 'steelblue', lw=2, label='Upper surface')
        ax_full.plot(x,       y_lower,       'coral',     lw=2, label='Lower surface')
        ax_full.fill_between(np.concatenate([x[::-1], x[1:]]), 
                             np.concatenate([y_upper[::-1], y_lower[1:]]), 
                             alpha=0.08, color='steelblue')
        ax_full.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
        ax_full.set_aspect('equal')
        ax_full.set_xlabel('x/c')
        ax_full.set_ylabel('y/c')
        ax_full.set_title(title_str, fontweight='bold', fontsize=11)
        ax_full.legend(fontsize=9, loc='upper right')
        ax_full.grid(True, alpha=0.3)

        # --- RIGHT: Leading Edge Detail ---
        mask = x <= 0.25
        ax_le.plot(x[::-1][x[::-1] <= 0.25], y_upper[::-1][x[::-1] <= 0.25], 'steelblue', lw=2.5)
        ax_le.plot(x[mask], y_lower[mask], 'coral', lw=2.5)
        ax_le.set_aspect('equal')
        ax_le.set_xlabel('x/c')
        ax_le.set_ylabel('y/c')
        ax_le.set_title(f'[{name}] Leading Edge Detail (x < 0.25)')
        ax_le.grid(True, alpha=0.3)

    # Adjust layout to fit the main title
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Multi-phase airfoil plot → {save_path}')

# ══════════════════════════════════════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_report(best_cst, best_cl, best_cd, best_cm, re, alpha,
                n_calls, n_feasible, eval_log, stability_label,
                cm_min, report_path, opt_mode='best_efficiency',
                cl_target=None):

    if opt_mode == 'target_lift':
        mode_line = (f'Optimization     : Target lift  '
                     f'CL = {cl_target} ± {CL_TOLERANCE}')
        objective_line = f'Objective        : Minimise CD  subject to CL ≈ {cl_target}'
    else:
        mode_line      = 'Optimization     : Best efficiency'
        objective_line = 'Objective        : Maximise CL/CD'

    lines = [
        'BAYESIAN OPTIMIZATION REPORT',
        '=' * 60,
        f'Flight condition  : Re={re:.0e}  alpha={alpha}°',
        f'Stability mode    : {stability_label}',
        f'CM constraint     : CM > {cm_min}',
        mode_line,
        objective_line,
        f'Iterations run    : {n_calls}',
        f'Feasible solutions: {n_feasible} / {n_calls}',
        '',
        'BEST AIRFOIL FOUND',
        '-' * 40,
        f'  CL       : {best_cl:.5f}',
    ]

    if opt_mode == 'target_lift' and cl_target is not None:
        diff = best_cl - cl_target
        lines.append(
            f'  CL target: {cl_target:.2f}  diff={diff:+.4f}  '
            f'{"✓ within tolerance" if abs(diff) <= CL_TOLERANCE else "⚠ outside tolerance"}'
        )

    lines += [
        f'  CD       : {best_cd:.6f}',
        f'  CM       : {best_cm:.5f}',
        f'  CL/CD    : {best_cl/best_cd:.3f}',
        f'  CM feasible: {"YES" if best_cm >= cm_min else "NO — constraint violated"}',
        '',
        'CST PARAMETERS',
        '-' * 40,
        '  Upper surface (cst_u0 to cst_u9):',
        f'    {[round(v,6) for v in best_cst[:10]]}',
        '  Lower surface (cst_l0 to cst_l9):',
        f'    {[round(v,6) for v in best_cst[10:]]}',
        '',
        'POPULATION STATISTICS (all feasible evaluations)',
        '-' * 40,
    ]

    feasible = [e for e in eval_log if e['feasible'] and not np.isnan(e.get('cl_cd', np.nan))]
    if feasible:
        cl_vals   = [e['cl']    for e in feasible]
        cd_vals   = [e['cd']    for e in feasible]
        clcd_vals = [e['cl_cd'] for e in feasible]
        lines += [
            f'  CL range   : [{min(cl_vals):.4f}, {max(cl_vals):.4f}]',
            f'  CD range   : [{min(cd_vals):.6f}, {max(cd_vals):.6f}]',
            f'  CL/CD range: [{min(clcd_vals):.2f}, {max(clcd_vals):.2f}]',
        ]

    lines += [
        '',
        'NEXT STEPS',
        '-' * 40,
        '  1. Validate best_airfoil.dat in XFOIL:',
        f'     XFOIL → LOAD best_airfoil.dat → OPER → VISC {int(re)} → ASEQ -5 15 1',
        '  2. Compare XFOIL CL/CD with optimizer prediction',
        f'     Expected CL≈{best_cl:.3f}  CD≈{best_cd:.5f}',
        '  3. If XFOIL result is within 10% → optimizer is reliable',
        '  4. If large discrepancy → retrain MLP on more airfoils first',
    ]

    text = '\n'.join(lines)
    with open(report_path, 'w') as f:
        f.write(text)
    print('\n' + text)
    print(f'\n  Report → {report_path}')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  SHAP SINGLE-PREDICTION EXPLANATION
# ══════════════════════════════════════════════════════════════════════════════

def explain_best_airfoil(model, scalers, best_cst, re_val, alpha_val,
                         best_cl, best_cd, best_cm, X_background,
                         stability_desc, save_path):
    """
    Run SHAP on the single best airfoil found by the optimizer.

    This answers the question: "WHY did the model predict these specific
    CL, CD, CM values for this geometry?"

    For each output (CL, CD, CM), it shows which CST parameters and
    flight conditions pushed the prediction above or below average —
    and by how much.

    Think of it like a doctor explaining a diagnosis:
    not just "your CL is 1.42" but "your CL is high BECAUSE
    your lower camber (CST 4) is aggressive (+0.18) AND you're
    at a favourable angle of attack (+0.31), but partially offset
    by your thick leading edge (-0.04)."

    Uses LinearExplainer on a small background set — runs in seconds,
    not minutes, because it is explaining ONE sample not 500.
    """
    print('\n  Running SHAP explanation on best airfoil...')

    cst_cols     = [f'cst_u{i}' for i in range(10)] + \
                   [f'cst_l{i}' for i in range(10)]
    feature_names = (
        [f'Upper CST {i}' for i in range(10)] +
        [f'Lower CST {i}' for i in range(10)] +
        ['Reynolds Number', 'Angle of Attack α']
    )

    # Build the single input vector for the best airfoil
    x_single = np.zeros((1, 22), dtype=np.float32)
    for i, col in enumerate(cst_cols):
        s = scalers[col]
        x_single[0, i] = (best_cst[i] - s['mean']) / (s['std'] + 1e-10)
    s = scalers['re']
    x_single[0, 20] = (re_val - s['mean']) / (s['std'] + 1e-10)
    s = scalers['alpha']
    x_single[0, 21] = (alpha_val - s['mean']) / (s['std'] + 1e-10)

    output_names  = ['CL (Lift)', 'CD (Drag)', 'CM (Moment)']
    output_colors = ['steelblue', 'coral', 'mediumseagreen']
    shap_results  = []

    # Compute SHAP values for each output separately
    for out_idx in range(3):
        def fn(X):
            with torch.no_grad():
                t   = torch.tensor(X, dtype=torch.float32)
                out = model(t).numpy()
            return out[:, out_idx]

        # KernelExplainer with small background (fast for single sample)
        explainer  = shap.KernelExplainer(fn, X_background[:50])
        sv         = explainer.shap_values(x_single, nsamples=50)
        shap_results.append(sv[0])   # shape (22,)

    # ── Build explanation plot ────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f'SHAP Explanation — Why This Airfoil?\n'
        f'Re={re_val:.0e}  α={alpha_val}°  '
        f'CL={best_cl:.4f}  CD={best_cd:.5f}  CM={best_cm:.4f}  '
        f'CL/CD={best_cl/best_cd:.1f}  |  {stability_desc}',
        fontsize=11, fontweight='bold', y=0.98
    )

    gs = plt.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # Top row: waterfall-style bar chart per output
    for col_idx, (sv, out_name, color) in enumerate(
            zip(shap_results, output_names, output_colors)):

        ax = fig.add_subplot(gs[0, col_idx])

        # Sort by absolute SHAP value, show top 10
        order    = np.argsort(np.abs(sv))[::-1][:10]
        vals     = sv[order]
        names    = [feature_names[i] for i in order]
        bar_cols = ['#2196F3' if v >= 0 else '#F44336' for v in vals]

        bars = ax.barh(range(len(vals)), vals, color=bar_cols,
                       edgecolor='white', height=0.7)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color='black', lw=1.0)
        ax.set_xlabel('SHAP value\n(blue = pushes up  |  red = pushes down)',
                      fontsize=8)
        ax.set_title(f'{out_name}', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.2, axis='x')

        # Annotate values
        for bar, val in zip(bars, vals):
            ax.text(val + (0.001 if val >= 0 else -0.001),
                    bar.get_y() + bar.get_height()/2,
                    f'{val:+.4f}', va='center', fontsize=7,
                    ha='left' if val >= 0 else 'right')

    # Bottom row: combined radar-style summary + plain language explanation
    # Left: All 22 features ranked by total |SHAP| across all outputs
    ax = fig.add_subplot(gs[1, :2])
    total_shap = np.abs(shap_results[0]) + \
                 np.abs(shap_results[1]) + \
                 np.abs(shap_results[2])
    order_all  = np.argsort(total_shap)[::-1][:12]
    vals_cl    = shap_results[0][order_all]
    vals_cd    = shap_results[1][order_all]
    vals_cm    = shap_results[2][order_all]
    names_all  = [feature_names[i] for i in order_all]

    x_pos  = np.arange(len(names_all))
    width  = 0.28
    ax.bar(x_pos - width, vals_cl, width, label='CL',
           color='steelblue', alpha=0.85, edgecolor='white')
    ax.bar(x_pos,          vals_cd, width, label='CD',
           color='coral',     alpha=0.85, edgecolor='white')
    ax.bar(x_pos + width,  vals_cm, width, label='CM',
           color='mediumseagreen', alpha=0.85, edgecolor='white')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names_all, rotation=35, ha='right', fontsize=8)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('SHAP value')
    ax.set_title('Top 12 Features — Effect on All Three Outputs',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    # Right: plain language summary box
    ax_text = fig.add_subplot(gs[1, 2])
    ax_text.axis('off')

    # Generate plain language lines from SHAP values
    def top_driver(sv, n=2):
        order = np.argsort(np.abs(sv))[::-1][:n]
        parts = []
        for i in order:
            direction = 'increases' if sv[i] > 0 else 'reduces'
            parts.append(f'{feature_names[i]} {direction} it ({sv[i]:+.3f})')
        return parts

    cl_drivers = top_driver(shap_results[0])
    cd_drivers = top_driver(shap_results[1])
    cm_drivers = top_driver(shap_results[2])

    summary_lines = [
        '── PLAIN LANGUAGE SUMMARY ──\n',
        f'CL = {best_cl:.4f}',
        f'  Main reason: {cl_drivers[0]}',
        f'  Also: {cl_drivers[1]}',
        '',
        f'CD = {best_cd:.5f}',
        f'  Main reason: {cd_drivers[0]}',
        f'  Also: {cd_drivers[1]}',
        '',
        f'CM = {best_cm:.4f}',
        f'  Main reason: {cm_drivers[0]}',
        f'  Also: {cm_drivers[1]}',
        '',
        f'CL/CD = {best_cl/best_cd:.2f}',
        '',
        '── GEOMETRY CHARACTER ──\n',
    ]

    # Characterise upper vs lower camber contribution
    upper_cl = sum(shap_results[0][i] for i in range(10))
    lower_cl = sum(shap_results[0][i+10] for i in range(10))
    if abs(lower_cl) > abs(upper_cl):
        summary_lines.append('Lift driven by lower camber shape.')
    else:
        summary_lines.append('Lift driven by upper surface shape.')

    upper_cd = sum(abs(shap_results[1][i]) for i in range(10))
    lower_cd = sum(abs(shap_results[1][i+10]) for i in range(10))
    if upper_cd > lower_cd * 1.3:
        summary_lines.append('Drag: upper surface boundary layer dominant.')
    elif lower_cd > upper_cd * 1.3:
        summary_lines.append('Drag: lower surface separation dominant.')
    else:
        summary_lines.append('Drag balanced across both surfaces.')

    cm_stable = best_cm >= -0.05
    summary_lines.append(
        f'Stability: {"✓ moment well-controlled" if cm_stable else "⚠ nose-down tendency"}'
    )

    ax_text.text(0.05, 0.97, '\n'.join(summary_lines),
                 transform=ax_text.transAxes,
                 fontsize=8.5, verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#f8f8f8',
                           edgecolor='#cccccc', alpha=0.9))

    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  SHAP explanation saved → {save_path}')

    # Save raw SHAP values as JSON for the dashboard geometry map
    shap_json_path = os.path.join(os.path.dirname(save_path), 'shap_values.json')
    try:
        shap_export = {
            'feature_names': feature_names,
            'shap_cl': [float(v) for v in shap_results[0]],
            'shap_cd': [float(v) for v in shap_results[1]],
            'shap_cm': [float(v) for v in shap_results[2]],
        }
        with open(shap_json_path, 'w') as f:
            json.dump(shap_export, f, indent=2)
        print(f'  SHAP values saved → {shap_json_path}')
    except Exception as e:
        print(f'  Warning: could not save shap_values.json — {e}')

    print(f'\n  Plain language summary:')
    for line in summary_lines:
        if line:
            print(f'    {line}')


def build_shap_background(scalers, data_csv, n=50):
    """
    Build a small normalized background matrix from the training set.
    Used as the SHAP reference distribution — 'what does an average
    airfoil look like at this stage of the pipeline?'
    50 samples is enough for KernelExplainer on a single prediction.
    """
    df       = pd.read_csv(data_csv)
    train_df = df[df['split'] == 'train'].sample(n=n, random_state=42)
    cst_cols = [f'cst_u{i}' for i in range(10)] + \
               [f'cst_l{i}' for i in range(10)]

    X = np.zeros((len(train_df), 22), dtype=np.float32)
    for i, col in enumerate(cst_cols):
        s = scalers[col]
        X[:, i] = (train_df[col].values - s['mean']) / (s['std'] + 1e-10)
    s = scalers['re']
    X[:, 20] = (train_df['re'].values - s['mean']) / (s['std'] + 1e-10)
    s = scalers['alpha']
    X[:, 21] = (train_df['alpha'].values - s['mean']) / (s['std'] + 1e-10)
    return X



# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-PHASE OPTIMIZATION — NEW FUNCTIONS (do not modify existing functions)
# ══════════════════════════════════════════════════════════════════════════════

# Reference scales for normalised cost terms — keeps CD, CL penalty, and
# CM penalty in the same order of magnitude so no term dominates the others.
_CD_REF  = 0.02    # typical good-airfoil CD
_CL_REF  = 0.05    # CL tolerance window
_CM_REF  = 0.05    # typical CM constraint margin


def parse_phases(phases_file):
    """
    Load and validate a flight profile JSON file.

    Expected structure:
    {
      "phases": [
        {"name": "climb",   "re": 500000,  "alpha": 8.0, "cl_target": 1.2,
         "stability": "B",  "cd_weight": 0.3},
        {"name": "cruise",  "re": 1000000, "alpha": 5.0, "cl_target": 0.8,
         "stability": "B",  "cd_weight": 1.0},
        {"name": "descent", "re": 300000,  "alpha": 3.0, "cl_target": 0.6,
         "stability": "C",  "cd_weight": 0.1}
      ],
      "phase_weights": [0.33, 0.34, 0.33]
    }

    Returns:
        phases       : list of dicts, each with resolved cm_min and cd_weight
        phase_weights: list of floats, normalised to sum to 1.0
    """
    if not os.path.exists(phases_file):
        print(f'\n  ERROR: phases file not found → {phases_file}')
        print(f'  Create it or use the default: flight_profile.json')
        sys.exit(1)

    with open(phases_file) as f:
        profile = json.load(f)

    required_phase_keys = {'name', 're', 'alpha', 'cl_target', 'stability'}
    phases = []
    for i, p in enumerate(profile.get('phases', [])):
        missing = required_phase_keys - set(p.keys())
        if missing:
            print(f'\n  ERROR: phase {i+1} missing keys: {missing}')
            sys.exit(1)

        stab_key = p['stability'].upper()
        if stab_key not in STABILITY_MODES:
            print(f'\n  ERROR: phase {i+1} stability "{p["stability"]}" invalid.')
            print(f'  Valid options: A, B, C, D')
            sys.exit(1)

        cm_min, stab_desc = STABILITY_MODES[stab_key]
        phases.append({
            'name'      : p['name'],
            're'        : float(p['re']),
            'alpha'     : float(p['alpha']),
            'cl_target' : float(p['cl_target']),
            'stability' : stab_key,
            'cm_min'    : cm_min,
            'stab_desc' : stab_desc,
            'cd_weight' : float(p.get('cd_weight', 0.5)),  # default 0.5 if omitted
        })

    if len(phases) < 2:
        print(f'\n  ERROR: multi_phase requires at least 2 phases, got {len(phases)}')
        sys.exit(1)

    # Phase weights — accept any positive values, normalise to sum=1.0
    raw_weights = profile.get('phase_weights', [1.0] * len(phases))
    if len(raw_weights) != len(phases):
        print(f'\n  ERROR: phase_weights length {len(raw_weights)} '
              f'does not match phases length {len(phases)}')
        sys.exit(1)

    total = sum(raw_weights)
    phase_weights = [w / total for w in raw_weights]

    return phases, phase_weights


def objective_multiphase(cst_params, phases, phase_weights):
    """
    Multi-phase objective function.

    Calls predict() ONCE PER PHASE per iteration — total 3 MLP calls.
    Computes a normalised composite cost across all phases.

    Per-phase cost (normalised so all terms are order-of-magnitude comparable):
        cost_i = cd_weight_i * (CD / CD_REF)
               + (CL_penalty / CL_REF)
               + (CM_penalty / CM_REF)

    Composite cost:
        J = sum_i( weight_i * cost_i )

    Feasible = ALL phases satisfy CL tolerance AND CM constraint.

    Why normalised:
        CD ≈ 0.015, raw CL penalty ≈ 8.0, raw CM penalty ≈ 0.025
        Without normalisation the CL penalty dominates and optimizer
        ignores CD and CM entirely. Normalisation keeps all three
        terms in the same order of magnitude.
    """
    global _model, _scalers, _device, _eval_log

    # ── Geometry validity — same pre-filter as single-phase ───────────────────
    valid, reason = _geometry_is_valid(cst_params)
    if not valid:
        _eval_log.append({
            'cl': np.nan, 'cd': np.nan, 'cm': np.nan,
            'cl_cd': np.nan, 'composite': 999.0, 'feasible': False,
            'invalid_geometry': reason,
        })
        return 999.0

    phase_results = []
    all_feasible  = True
    composite     = 0.0

    for phase, weight in zip(phases, phase_weights):
        cl, cd, cm = predict(
            _model, _scalers, cst_params,
            phase['re'], phase['alpha'], _device
        )

        if cd <= 0 or any(np.isnan(v) for v in [cl, cd, cm]):
            # Physically impossible prediction — penalise heavily
            phase_results.append({
                'name': phase['name'], 'cl': np.nan, 'cd': np.nan,
                'cm': np.nan, 'feasible': False, 'cost': 999.0
            })
            composite    += weight * 999.0
            all_feasible  = False
            continue

        # ── CL penalty ────────────────────────────────────────────────────────
        cl_violation = max(0.0, abs(cl - phase['cl_target']) - CL_TOLERANCE)
        cl_penalty   = CM_PENALTY * 20 * (cl_violation ** 2)   # consistent with single-phase

        # ── CM penalty ────────────────────────────────────────────────────────
        cm_violation = max(0.0, phase['cm_min'] - cm)
        cm_penalty   = CM_PENALTY * (cm_violation ** 2)

        # ── Normalised cost ───────────────────────────────────────────────────
        cd_term  = phase['cd_weight'] * (cd / _CD_REF)
        cl_term  = cl_penalty / _CL_REF
        cm_term  = cm_penalty / _CM_REF
        cost_i   = cd_term + cl_term + cm_term

        phase_feasible = (
            abs(cl - phase['cl_target']) <= CL_TOLERANCE * 2  # relaxed for multi-phase
            and cm >= phase['cm_min']
        )
        if not phase_feasible:
            all_feasible = False

        composite += weight * cost_i
        phase_results.append({
            'name'     : phase['name'],
            'cl'       : cl,
            'cd'       : cd,
            'cm'       : cm,
            'cl_target': phase['cl_target'],
            'cm_min'   : phase['cm_min'],
            'feasible' : phase_feasible,
            'cost'     : cost_i,
            'cl_miss'  : cl - phase['cl_target'],
        })

    _eval_log.append({
        'cst_params'   : list(cst_params),
        'composite'    : composite,
        'feasible'     : all_feasible,
        'phase_results': phase_results,
        # Convenience fields for convergence plotting
        'cl'  : phase_results[0]['cl']  if phase_results else np.nan,
        'cd'  : phase_results[0]['cd']  if phase_results else np.nan,
        'cm'  : phase_results[0]['cm']  if phase_results else np.nan,
        'cl_cd': (phase_results[0]['cl'] / phase_results[0]['cd']
                  if phase_results and phase_results[0]['cd'] > 0 else np.nan),
    })

    return composite


def seed_multiphase(phases, bounds, cst_cols, df_train):
    """
    Build seed pool for multi-phase optimization.
    Strategy: MLP-scored population approach per phase — same logic as single-phase.
      1. Filter by Re (closest), alpha +-2deg, CM constraint, CL +-20%.
      2. Score all candidates through MLP at exact phase conditions.
      3. Rank by MLP CL/CD — best efficiency first.
      4. Take top 8 per phase, deduplicated across phases.
    No LHS random explorers — they produce symmetric/unphysical shapes.
    """
    seen_airfoils = set()
    all_candidates_by_phase = []
    CL_WINDOW_SEED = 0.20

    for phase in phases:
        re_vals    = df_train['re'].unique()
        closest_re = re_vals[np.argmin(np.abs(re_vals - phase['re']))]
        nearby     = df_train[
            (df_train['re'] == closest_re) &
            (df_train['alpha'].between(phase['alpha'] - 2, phase['alpha'] + 2)) &
            (df_train['cm'] >= phase['cm_min'])
        ].copy()
        if len(nearby) == 0:
            nearby = df_train[
                (df_train['re'] == closest_re) &
                (df_train['cm'] >= phase['cm_min'])
            ].copy()

        cl_lo = phase['cl_target'] * (1 - CL_WINDOW_SEED)
        cl_hi = phase['cl_target'] * (1 + CL_WINDOW_SEED)
        candidates = nearby[nearby['cl'].between(cl_lo, cl_hi)].copy()
        if len(candidates) < 5:
            cl_lo = phase['cl_target'] * (1 - CL_WINDOW_SEED * 2)
            cl_hi = phase['cl_target'] * (1 + CL_WINDOW_SEED * 2)
            candidates = nearby[nearby['cl'].between(cl_lo, cl_hi)].copy()

        if len(candidates) == 0:
            continue
        if 'airfoil' in candidates.columns:
            candidates = candidates.drop_duplicates(subset='airfoil')

        # MLP re-score at exact phase conditions
        mlp_cls, mlp_cds, mlp_clcds = [], [], []
        for _, row in candidates.iterrows():
            cst_vec = [row[c] for c in cst_cols]
            p_cl, p_cd, _ = predict(_model, _scalers, cst_vec,
                                    phase['re'], phase['alpha'], _device)
            p_clcd = p_cl / p_cd if p_cd > 0 else 0.0
            mlp_cls.append(p_cl)
            mlp_cds.append(p_cd)
            mlp_clcds.append(p_clcd)

        candidates = candidates.copy()
        candidates['mlp_cl']      = mlp_cls
        candidates['mlp_cd']      = mlp_cds
        candidates['mlp_clcd']    = mlp_clcds
        candidates['mlp_cl_dist'] = (candidates['mlp_cl'] - phase['cl_target']).abs()

        mlp_ok = candidates[
            candidates['mlp_cl'].between(phase['cl_target'] * 0.70,
                                         phase['cl_target'] * 1.30)
        ].copy()
        if len(mlp_ok) >= 5:
            candidates = mlp_ok

        candidates = candidates.sort_values('mlp_clcd', ascending=False)

        best = candidates.iloc[0]
        print(f'  [{phase["name"]:<10}] {len(candidates)} candidates  '
              f'best MLP CL/CD={best["mlp_clcd"]:.1f}  '
              f'({best.get("airfoil","?")})')

        all_candidates_by_phase.append((phase['name'], candidates))

    # Collect top-8 per phase, deduplicated globally
    anchor_csts = []
    for phase_name, cands in all_candidates_by_phase:
        count = 0
        for _, row in cands.iterrows():
            if count >= 8:
                break
            name = row.get('airfoil', '')
            if name and name in seen_airfoils:
                continue
            anchor_csts.append([row[c] for c in cst_cols])
            if name:
                seen_airfoils.add(name)
            count += 1

    print(f'\n  Total seeds collected: {len(anchor_csts)} '
          f'(deduplicated across {len(phases)} phases)')

    return anchor_csts


def report_multiphase(best_cst, best_eval, phases, phase_weights,
                       n_calls, n_feasible, save_path):
    """
    Write multi-phase optimization report.

    Shows per-phase CL, CD, CM and CL deviation so the designer sees
    exactly which phase the compromise hurt most.
    """
    phase_results = best_eval['phase_results']

    lines = [
        'MULTI-PHASE OPTIMIZATION REPORT',
        '=' * 65,
        '',
        f'Phases run       : {len(phases)}',
        f'Iterations       : {n_calls}',
        f'Feasible results : {n_feasible}',
        f'Composite score  : {best_eval["composite"]:.6f}  (lower = better)',
        '',
        'PHASE WEIGHTS (normalised)',
        '-' * 65,
    ]
    for phase, w in zip(phases, phase_weights):
        lines.append(f'  {phase["name"]:<12} weight={w:.3f}  '
                     f'cd_weight={phase["cd_weight"]}  '
                     f'stability=[{phase["stability"]}] CM>{phase["cm_min"]}')

    lines += [
        '',
        'PER-PHASE RESULTS — Best CST Found',
        '-' * 65,
        f'{"Phase":<12} {"CL tgt":>8} {"CL act":>8} {"CL miss":>8} '
        f'{"CD":>9} {"CM":>9} {"Feasible":>9}',
        '-' * 65,
    ]

    for pr in phase_results:
        cl_miss = pr.get('cl_miss', np.nan)
        tick    = '✓' if pr['feasible'] else '⚠'
        lines.append(
            f'  {pr["name"]:<10} '
            f'{pr["cl_target"]:>8.3f} '
            f'{pr["cl"]:>8.4f} '
            f'{cl_miss:>+8.4f} '
            f'{pr["cd"]:>9.5f} '
            f'{pr["cm"]:>9.5f} '
            f'{"  "+tick:>9}'
        )

    lines += [
        '-' * 65,
        '',
        'PHASE CONFLICT SUMMARY',
        '-' * 65,
    ]
    for pr in phase_results:
        if not pr['feasible']:
            cl_miss = abs(pr.get('cl_miss', 0))
            cm_ok   = pr['cm'] >= pr.get('cm_min', -9999)
            cl_ok   = cl_miss <= CL_TOLERANCE
            reason  = []
            if not cl_ok:
                reason.append(f'CL miss={cl_miss:.4f} > tolerance={CL_TOLERANCE}')
            if not cm_ok:
                reason.append(f'CM={pr["cm"]:.4f} < threshold={pr.get("cm_min")}')
            lines.append(f'  ⚠ {pr["name"]}: {" | ".join(reason)}')
        else:
            lines.append(f'  ✓ {pr["name"]}: all constraints satisfied')

    lines += [
        '',
        'CST PARAMETERS',
        '-' * 65,
        '  Upper surface:',
        f'    {[round(v,6) for v in best_cst[:10]]}',
        '  Lower surface:',
        f'    {[round(v,6) for v in best_cst[10:20]]}',
        '',
        'NOTE: SHAP explanation used cruise phase conditions as reference.',
        '      Validate best_airfoil.dat in XFOIL at each phase condition.',
    ]

    text = '\n'.join(lines)
    with open(save_path, 'w') as f:
        f.write(text)
    print(text)
    print(f'\n  Report → {save_path}')


def _run_multiphase(args, n_calls):
    """
    Multi-phase optimization entry point.
    Called from main() when --mode multi_phase is passed.
    All single-phase functions are completely untouched.
    """
    global _model, _scalers, _device, _eval_log, OUT_DIR

    # Per-run output directory — created before any file writes
    OUT_DIR = _setup_run_dir('multi_phase')
    print(f'\n  Mode: MULTI-PHASE OPTIMIZATION')
    print(f'  Run folder: optimizer_results/{os.path.basename(OUT_DIR)}')
    print(f'  Phases file: {args.phases_file}')

    phases, phase_weights = parse_phases(args.phases_file)

    print(f'\n  {"─"*60}')
    print(f'  FLIGHT PROFILE')
    print(f'  {"─"*60}')
    for phase, w in zip(phases, phase_weights):
        print(f'  [{phase["name"]:<10}]  '
              f'Re={phase["re"]:.0e}  alpha={phase["alpha"]}°  '
              f'CL={phase["cl_target"]}  stability=[{phase["stability"]}]  '
              f'cd_weight={phase["cd_weight"]}  weight={w:.3f}')
    print(f'  {"─"*60}')
    print(f'  Phase weights normalised to: {[round(w,3) for w in phase_weights]}')

    device    = torch.device('cpu')
    _device   = device
    _eval_log = []

    print('\n  Loading model and scalers...')
    _model, cfg = load_model(MODEL_PATH, device)
    _scalers    = load_scalers(SCALER_PATH)

    print('\n  Running pre-flight checks per phase...')
    df_check   = pd.read_csv(DATA_CSV)
    n_conflicts = 0

    for phase in phases:
        re_vals    = df_check['re'].unique()
        closest_re = re_vals[np.argmin(np.abs(re_vals - phase['re']))]
        nearby     = df_check[
            (df_check['re'] == closest_re) &
            (df_check['alpha'].between(phase['alpha'] - 2, phase['alpha'] + 2))
        ]
        cl_feasible = nearby[
            nearby['cl'].between(phase['cl_target'] - CL_TOLERANCE,
                                  phase['cl_target'] + CL_TOLERANCE)
        ]
        both_ok = cl_feasible[cl_feasible['cm'] >= phase['cm_min']]

        if len(cl_feasible) == 0:
            print(f'  ⚠ [{phase["name"]}]: CL={phase["cl_target"]} never seen '
                  f'at Re={phase["re"]:.0e} alpha={phase["alpha"]}° — will extrapolate')
            n_conflicts += 1
        elif len(both_ok) == 0:
            best_cm = cl_feasible['cm'].max()
            print(f'  ⚠ [{phase["name"]}]: CL={phase["cl_target"]} achievable '
                  f'but best CM={best_cm:.4f} < threshold={phase["cm_min"]}')
            print(f'     Optimizer will try — expect fewer feasible results on this phase')
            n_conflicts += 1
        else:
            print(f'  ✓ [{phase["name"]}]: {len(both_ok)} training examples satisfy all constraints')

    if n_conflicts == len(phases):
        print(f'\n  ✗ ALL {len(phases)} phases conflict. Aborting.')
        print(f'  Relax stability modes or adjust CL targets in {args.phases_file}')
        return
    elif n_conflicts > 0:
        print(f'\n  {n_conflicts}/{len(phases)} phases have conflicts — proceeding with caution.')

    # ── Geometric consistency check ───────────────────────────────────────────
    # Warn if the CL targets are geometrically inconsistent with a linear polar.
    # One airfoil cannot hit all targets exactly — multi-phase finds the best
    # compromise. This check shows the user what the natural polar implies.
    print('\n  Geometric consistency check...')
    if len(phases) >= 2:
        ph0, ph1 = phases[0], phases[1]
        if ph1['alpha'] != ph0['alpha']:
            cl_alpha = (ph1['cl_target'] - ph0['cl_target']) / (ph1['alpha'] - ph0['alpha'])
            cl0      = ph0['cl_target'] - cl_alpha * ph0['alpha']
            print(f'  Linear polar implied by {ph0["name"]} + {ph1["name"]}:')
            print(f'    CLα = {cl_alpha:.4f}/deg   CL0 = {cl0:.4f}')
            any_inconsistent = False
            for ph in phases[2:]:
                cl_pred = cl0 + cl_alpha * ph['alpha']
                miss    = cl_pred - ph['cl_target']
                status  = '✓' if abs(miss) <= CL_TOLERANCE else '⚠'
                print(f'    {ph["name"]}: predicted CL={cl_pred:.3f}  '
                      f'target={ph["cl_target"]}  miss={miss:+.3f}  {status}')
                if abs(miss) > CL_TOLERANCE:
                    any_inconsistent = True
                    print(f'      → Miss exceeds tolerance ±{CL_TOLERANCE}.')
                    print(f'        Optimizer will compromise — '
                          f'consider adjusting {ph["name"]} CL to ≈{cl_pred:.2f}')
            if not any_inconsistent:
                print(f'  ✓ All phase CL targets are geometrically consistent.')
            else:
                print(f'\n  NOTE: Inconsistent targets are normal for multi-phase design.')
                print(f'  The optimizer minimises total weighted miss — not exact satisfaction.')
                print(f'  Check CL miss per phase in the final report.')

    print('\n  Setting up search space...')
    _, cst_cols = compute_cst_bounds(DATA_CSV)

    print('  Building multi-phase seed pool...')
    df_train = pd.read_csv(DATA_CSV)
    df_train = df_train[df_train['split'] == 'train'].copy()
    df_train['cl_cd'] = df_train['cl'] / df_train['cd']

    # Seeds first — bounds derived from seeds (tighter, population-anchored)
    raw_seeds = seed_multiphase(phases, None, cst_cols, df_train)

    EXPLORE_MARGIN_MP = 0.30
    if len(raw_seeds) >= 2:
        seed_arr = np.array(raw_seeds[:10])
        bounds = []
        for j in range(len(cst_cols)):
            col_vals = seed_arr[:, j]
            lo   = col_vals.min()
            hi   = col_vals.max()
            span = max(hi - lo, 0.05)
            bounds.append((float(lo - EXPLORE_MARGIN_MP * span),
                           float(hi + EXPLORE_MARGIN_MP * span)))
        lo_arr_d = np.array([b[0] for b in bounds])
        hi_arr_d = np.array([b[1] for b in bounds])
        print(f'  Bounds: top-10 seed range +- {EXPLORE_MARGIN_MP*100:.0f}%  '
              f'mean span={float((hi_arr_d-lo_arr_d).mean()):.4f}')
    else:
        bounds, _ = compute_cst_bounds(DATA_CSV, margin=0.15)
        print(f'  Bounds: global (few seeds found)')

    space = [Real(lo, hi, name=col) for (lo, hi), col in zip(bounds, cst_cols)]
    lo_arr = np.array([b[0] for b in bounds])
    hi_arr = np.array([b[1] for b in bounds])
    x0 = [list(np.clip(xp, lo_arr, hi_arr)) for xp in raw_seeds]
    y0 = [objective_multiphase(xp, phases, phase_weights) for xp in x0]

    n_feasible_seed = sum(1 for e in _eval_log if e['feasible'])
    print(f'  Seeds: {len(x0)} total  |  {n_feasible_seed} fully feasible at seed stage')

    print(f'\n  Running multi-phase Bayesian optimization ({n_calls} iterations)...')
    print(f'  MLP called {len(phases)}× per iteration = {n_calls * len(phases)} total predictions')
    print(f'  Progress every 10 iterations.\n')

    iteration_counter     = [0]

    def callback_mp(res):
        iteration_counter[0] += 1
        if iteration_counter[0] % 10 == 0:
            feasible_so_far = [e for e in _eval_log
                               if e['feasible'] and not np.isnan(e.get('composite', np.nan))]
            if feasible_so_far:
                best = min(feasible_so_far, key=lambda e: e['composite'])
                print(f'  iter {iteration_counter[0]:>4} | '
                      f'composite={best["composite"]:.5f} | '
                      f'feasible={len(feasible_so_far)}')
            else:
                print(f'  iter {iteration_counter[0]:>4} | no feasible result yet')

        # Write progress for dashboard polling
        try:
            n_feas   = sum(1 for e in _eval_log if e['feasible'])
            best_c   = min((e['composite'] for e in _eval_log
                            if e['feasible'] and not np.isnan(e.get('composite', np.nan))),
                           default=999.0)
            with open(os.path.join(OUT_DIR, 'progress.txt'), 'w') as pf:
                pf.write(f'{iteration_counter[0]},{n_calls},{n_feas},{best_c:.5f},multi\n')
        except Exception:
            pass

    gp_minimize(
        func             = lambda x: objective_multiphase(x, phases, phase_weights),
        dimensions       = space,
        n_calls          = n_calls,
        n_initial_points = 25,
        x0               = x0,
        y0               = y0,
        acq_func         = 'LCB',
        noise            = 0.01,
        random_state     = 42,
        callback         = callback_mp,
        verbose          = False,
    )

    feasible_evals = [e for e in _eval_log
                      if e['feasible'] and not np.isnan(e.get('composite', np.nan))]

    if not feasible_evals:
        valid = [e for e in _eval_log if not np.isnan(e.get('composite', np.nan))]
        if not valid:
            print('  ✗ No valid results. Try more iterations or relax constraints.')
            return
        best_eval = min(valid, key=lambda e: e['composite'])
        print('  ⚠ No fully feasible result — returning best composite.')
    else:
        best_eval = min(feasible_evals, key=lambda e: e['composite'])

    best_cst   = best_eval['cst_params']
    n_feasible = len(feasible_evals)

    print(f'\n  {"═"*65}')
    print(f'  MULTI-PHASE BEST RESULT')
    print(f'  {"═"*65}')
    print(f'  {"Phase":<12} {"CL tgt":>8} {"CL act":>8} {"CL miss":>8} '
          f'{"CD":>9} {"CM":>9} {"OK":>4}')
    print(f'  {"─"*65}')
    for pr in best_eval['phase_results']:
        miss = pr.get('cl_miss', np.nan)
        tick = '✓' if pr['feasible'] else '⚠'
        print(f'  {pr["name"]:<12} {pr["cl_target"]:>8.3f} {pr["cl"]:>8.4f} '
              f'{miss:>+8.4f} {pr["cd"]:>9.5f} {pr["cm"]:>9.5f} {tick:>4}')
    print(f'  {"─"*65}')
    print(f'  Composite: {best_eval["composite"]:.6f}  |  '
          f'Feasible: {n_feasible}/{n_calls + len(x0)}')
    print(f'  {"═"*65}')

    coords, x_arr, y_upper, y_lower = cst_to_coordinates(best_cst)
    dat_path = os.path.join(OUT_DIR, 'best_airfoil.dat')
    save_dat_file(coords, 'opt_multiphase', dat_path)

    cruise_pr = next((p for p in best_eval['phase_results']
                      if 'crui' in p['name'].lower()), best_eval['phase_results'][0])
    cruise_ph = next((p for p in phases if 'crui' in p['name'].lower()), phases[0])
    plot_airfoil(
        x_arr, y_upper, y_lower,
        cruise_pr['cl'], cruise_pr['cd'], cruise_pr['cm'],
        cruise_ph['re'], cruise_ph['alpha'],
        os.path.join(OUT_DIR, 'best_airfoil.png')
    )

    # NEW BLOCK:
    plot_airfoil_multiphase_stacked(
        x_arr, y_upper, y_lower,
        best_eval['phase_results'],
        phases,
        os.path.join(OUT_DIR, 'best_airfoil_all_phases.png')
    )

    eval_rows = []
    for e in _eval_log:
        row = {'composite': e['composite'], 'feasible': e['feasible']}
        for pr in e.get('phase_results', []):
            row[f'{pr["name"]}_cl'] = pr['cl']
            row[f'{pr["name"]}_cd'] = pr['cd']
            row[f'{pr["name"]}_cm'] = pr['cm']
        eval_rows.append(row)
    pd.DataFrame(eval_rows).to_csv(
        os.path.join(OUT_DIR, 'all_evaluations.csv'), index=False)

    report_multiphase(
        best_cst, best_eval, phases, phase_weights,
        n_calls + len(x0), n_feasible,
        os.path.join(OUT_DIR, 'optimization_report.txt')
    )

    if SHAP_AVAILABLE:
        X_bg = build_shap_background(_scalers, DATA_CSV, n=50)
        explain_best_airfoil(
            _model, _scalers, best_cst,
            cruise_ph['re'], cruise_ph['alpha'],
            cruise_pr['cl'], cruise_pr['cd'], cruise_pr['cm'],
            X_bg,
            f'[{cruise_ph["stability"]}] cruise reference',
            os.path.join(OUT_DIR, 'shap_explanation.png')
        )

    print('\n' + '█'*65)
    print('  MULTI-PHASE DONE.')
    print(f'  Best airfoil → {dat_path}')
    print(f'  Validate in XFOIL at each phase condition.')
    print('█'*65 + '\n')


def main():
    global _model, _scalers, _device, _re, _alpha, _cm_min, _eval_log

    # ── Parse arguments ───────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description='Airfoil Bayesian Optimizer')
    parser.add_argument('--re',        type=float, default=DEFAULT_RE,
                        help=f'Reynolds number (default: {DEFAULT_RE})')
    parser.add_argument('--alpha',     type=float, default=DEFAULT_ALPHA,
                        help=f'Angle of attack in degrees (default: {DEFAULT_ALPHA})')
    parser.add_argument('--n_calls',   type=int,   default=DEFAULT_NCALLS,
                        help=f'Optimization iterations (default: {DEFAULT_NCALLS})')
    parser.add_argument('--stability', type=str,   default=None,
                        choices=['A','B','C','D'],
                        help='Stability mode A/B/C/D (interactive if not set)')
    parser.add_argument('--cl_target',   type=float, default=None,
                        help='Target CL for target_lift mode (e.g. 0.8). '
                             'If omitted, runs best_efficiency mode.')
    parser.add_argument('--mode',        type=str,   default=None,
                        choices=['target_lift', 'best_efficiency', 'multi_phase'],
                        help='Optimization mode. multi_phase requires --phases_file.')
    parser.add_argument('--phases_file', type=str,   default='flight_profile.json',
                        help='JSON file for multi_phase mode (default: flight_profile.json)')
    args = parser.parse_args()

    re_val    = args.re
    alpha_val = args.alpha
    n_calls   = args.n_calls

    print('\n' + '█'*65)
    print('  AIRFOIL BAYESIAN OPTIMIZER')
    print('█'*65)

    # ── Multi-phase early intercept ───────────────────────────────────────────
    # If --mode multi_phase is requested, hand off to multi-phase flow entirely.
    # Single-phase modes (target_lift, best_efficiency) continue below unchanged.
    if args.mode == 'multi_phase':
        _run_multiphase(args, n_calls)
        return
    # If not provided via CLI, prompt interactively.
    # This is the user-facing abstraction over raw CM values.
    if args.stability:
        stability_key = args.stability.upper()
    else:
        print('\n  Select stability mode for your aircraft:\n')
        for key, (cm_thresh, desc) in STABILITY_MODES.items():
            print(f'    [{key}]  {desc}')
        print()
        while True:
            choice = input('  Enter choice (A/B/C/D) [default B]: ').strip().upper()
            if choice == '':
                choice = DEFAULT_STABILITY
            if choice in STABILITY_MODES:
                stability_key = choice
                break
            print('  Invalid choice — enter A, B, C, or D')

    cm_min_val, stability_desc = STABILITY_MODES[stability_key]
    _cm_min = cm_min_val

    # ── Optimization mode selection ───────────────────────────────────────────
    # If --cl_target was passed via CLI, use target_lift mode directly.
    # Otherwise prompt interactively.
    if args.cl_target is not None:
        cl_target_val = args.cl_target
        opt_mode      = 'target_lift'
    else:
        print('\n  Optimization mode:\n')
        print('    [1]  Target lift  — I need a specific CL, find minimum CD')
        print('    [2]  Best efficiency — find the best CL/CD freely\n')
        while True:
            mode_choice = input('  Enter choice (1/2) [default 2]: ').strip()
            if mode_choice == '' or mode_choice == '2':
                opt_mode      = 'best_efficiency'
                cl_target_val = None
                break
            elif mode_choice == '1':
                opt_mode = 'target_lift'
                while True:
                    raw = input('  Enter CL target (range -0.5 to 2.0, e.g. 0.8): ').strip()
                    try:
                        cl_target_val = float(raw)
                        if -0.5 <= cl_target_val <= 2.0:
                            break
                        print('  Out of range — enter a value between -0.5 and 2.0')
                    except ValueError:
                        print('  Invalid — enter a number like 0.8 or 1.2')
                break
            else:
                print('  Enter 1 or 2')

    _cl_target = cl_target_val
    _opt_mode  = opt_mode

    # Print confirmed setup
    print(f'\n  Flight condition : Re={re_val:.0e}  alpha={alpha_val}°')
    print(f'  Stability mode   : [{stability_key}] {stability_desc}')
    print(f'  CM constraint    : CM > {cm_min_val}')
    if opt_mode == 'target_lift':
        print(f'  Optimization     : Target lift  CL = {cl_target_val} ± {CL_TOLERANCE}')
        print(f'  Objective        : Minimise CD subject to CL ≈ {cl_target_val}')
    else:
        print(f'  Optimization     : Best efficiency')
        print(f'  Objective        : Maximise CL/CD')
    print(f'  Iterations       : {n_calls}')
    print(f'  Thickness limit  : {MIN_THICKNESS*100:.1f}% min  —  {MAX_THICKNESS*100:.0f}% max chord')

    # ── Per-run output directory ──────────────────────────────────────────────
    global OUT_DIR
    tag    = f'single_Re{int(re_val)}_a{alpha_val}'
    if opt_mode == 'target_lift':
        tag += f'_CL{cl_target_val}'
    OUT_DIR = _setup_run_dir(tag)
    print(f'  Run folder       : optimizer_results/{os.path.basename(OUT_DIR)}')

    # ── Setup ─────────────────────────────────────────────────────────────────
    device    = torch.device('cpu')
    _device   = device
    _re       = re_val
    _alpha    = alpha_val
    _eval_log = []
    # _cm_min, _cl_target, _opt_mode already set above

    print('\n  Loading model and scalers...')
    _model, cfg = load_model(MODEL_PATH, device)
    _scalers    = load_scalers(SCALER_PATH)

    # ── Pre-flight compatibility check ────────────────────────────────────────
    # Before running 150 iterations, scan the training data to verify that
    # the requested CL_target is physically achievable under the chosen
    # stability mode at the requested Re and alpha.
    #
    # Why this matters:
    # High lift (CL > 0.9) requires camber. Camber creates negative CM.
    # If stability mode is strict (CM > -0.05) and CL target is high,
    # these two constraints are physically incompatible — no airfoil in
    # your training distribution satisfies both simultaneously.
    # Detecting this upfront saves 15 minutes of wasted optimization.
    if _opt_mode == 'target_lift':
        print('\n  Running pre-flight compatibility check...')
        df_check = pd.read_csv(DATA_CSV)

        # Find training rows near requested Re and alpha
        re_tol    = df_check['re'].unique()
        closest_re = re_tol[np.argmin(np.abs(re_tol - re_val))]
        nearby = df_check[
            (df_check['re'] == closest_re) &
            (df_check['alpha'].between(alpha_val - 2, alpha_val + 2))
        ]

        if len(nearby) > 0:
            # Check how many rows hit CL_target within tolerance
            cl_feasible = nearby[
                nearby['cl'].between(cl_target_val - CL_TOLERANCE,
                                     cl_target_val + CL_TOLERANCE)
            ]
            # Among those, how many also satisfy CM constraint
            both_feasible = cl_feasible[cl_feasible['cm'] >= cm_min_val]

            n_cl_only  = len(cl_feasible)
            n_both     = len(both_feasible)
            n_nearby   = len(nearby)

            print(f'  Training rows near Re={closest_re:.0e}, alpha=±2° of {alpha_val}°: {n_nearby}')
            print(f'  Rows with CL ≈ {cl_target_val} (±{CL_TOLERANCE}): {n_cl_only}')
            print(f'  Rows satisfying BOTH CL target AND CM > {cm_min_val}: {n_both}')

            if n_cl_only == 0:
                print(f'\n  ⚠ WARNING: CL = {cl_target_val} was never observed at these')
                print(f'  conditions in your training data.')
                print(f'  The optimizer will extrapolate — results may be unreliable.')
                print(f'  Consider: lower CL target, higher alpha, or higher Re.')

            elif n_both == 0 and n_cl_only > 0:
                # CL is achievable but CM constraint blocks everything
                median_cm         = cl_feasible['cm'].median()
                best_cm_available = cl_feasible['cm'].max()   # best case in training data
                suggested_mode    = None
                for mode_key, (cm_thresh, _) in STABILITY_MODES.items():
                    if best_cm_available >= cm_thresh:
                        suggested_mode = mode_key
                        break

                # HARD CONFLICT: even the best airfoil in training data violates CM
                # e.g. every airfoil at CL=1.1 has CM < -0.10 — truly incompatible
                # SOFT WARNING: median violates but best case does not
                # e.g. NACA 23012 type — reflexed camber achieves high CL with low |CM|
                # These exist in training data so the optimizer should try

                # Compute alpha alternatives and max CL under this mode
                alpha_needed  = df_check[df_check['re'] == closest_re].copy()
                cl_at_alphas  = alpha_needed.groupby('alpha')['cl'].median()
                viable_alphas = cl_at_alphas[
                    cl_at_alphas.between(cl_target_val - CL_TOLERANCE,
                                         cl_target_val + CL_TOLERANCE)
                ].index.tolist()
                mode_rows      = df_check[
                    (df_check['re'] == closest_re) &
                    (df_check['cm'] >= cm_min_val)
                ]
                max_cl_in_mode = mode_rows['cl'].max() if len(mode_rows) > 0 else 0.0

                if best_cm_available < cm_min_val:
                    # ── HARD CONFLICT ─────────────────────────────────────────
                    # Even the best airfoil in your entire training data at this
                    # CL target violates the CM constraint. Truly incompatible.
                    print(f'\n  {"█"*60}')
                    print(f'  ⚠  PHYSICAL CONFLICT — CANNOT PROCEED')
                    print(f'  {"█"*60}')
                    print(f'')
                    print(f'  You requested : CL = {cl_target_val}  stability [{stability_key}] (CM > {cm_min_val})')
                    print(f'  The problem   : No airfoil in training data satisfies both.')
                    print(f'  Best CM found at CL ≈ {cl_target_val}: {best_cm_available:.4f} — still below {cm_min_val}')
                    print(f'  Median CM at this CL: {median_cm:.4f}')
                    print(f'')
                    print(f'  YOUR OPTIONS:')
                    print(f'')
                    if suggested_mode and suggested_mode != stability_key:
                        sug_desc = STABILITY_MODES[suggested_mode][1]
                        print(f'  Option 1 — Relax stability (recommended):')
                        print(f'    → python bayesian_optimizer.py --stability {suggested_mode} --cl_target {cl_target_val} --re {int(re_val)} --alpha {alpha_val}')
                        print(f'')
                    print(f'  Option 2 — Lower CL target (max feasible ≈ {max_cl_in_mode:.2f}):')
                    print(f'    → python bayesian_optimizer.py --stability {stability_key} --cl_target {max_cl_in_mode:.2f} --re {int(re_val)} --alpha {alpha_val}')
                    print(f'')
                    if viable_alphas:
                        best_alpha = min(viable_alphas, key=lambda a: abs(a - alpha_val))
                        print(f'  Option 3 — Increase alpha to {best_alpha}°:')
                        print(f'    → python bayesian_optimizer.py --stability {stability_key} --cl_target {cl_target_val} --re {int(re_val)} --alpha {best_alpha}')
                        print(f'')
                    print(f'  Optimization aborted. No iterations wasted.')
                    print(f'  {"█"*60}\n')
                    return

                else:
                    # ── SOFT WARNING ──────────────────────────────────────────
                    # Median CM violates constraint BUT best case does not.
                    # Reflexed/S-camber airfoils (e.g. NACA 23012 type) exist
                    # in training data that achieve high CL with controlled CM.
                    # The optimizer should search for them — just warn it is rare.
                    pct = (n_both / n_cl_only * 100) if n_cl_only > 0 else 0
                    print(f'\n  ⚠ NOTE: CL = {cl_target_val} with CM > {cm_min_val} is rare but achievable.')
                    print(f'  Only {n_both}/{n_cl_only} ({pct:.0f}%) training airfoils at this CL satisfy CM constraint.')
                    print(f'  Best CM seen at CL ≈ {cl_target_val}: {best_cm_available:.4f}  ← reflexed camber type')
                    print(f'  The optimizer will search harder for these shapes.')
                    print(f'  Expect fewer feasible results and longer convergence.')
                    print(f'  Proceeding...\n')

            else:
                print(f'  ✓ CL target and stability mode are compatible. Proceeding.')
        else:
            print(f'  ⚠ No training data near these conditions — proceeding with caution.')

    # ══════════════════════════════════════════════════════════════════════════
    # MANIFOLD-ANCHORED SEARCH
    # ══════════════════════════════════════════════════════════════════════════
    #
    # Core idea: never wander into regions of CST space where the MLP
    # has no training data. Instead:
    #
    #   1. Filter the training CSV to find real airfoils that already
    #      achieve something close to your target condition.
    #      These are called CANDIDATES.
    #
    #   2. Derive search bounds from CST coefficients of those candidates.
    #      Add a small EXPLORE_MARGIN so the optimizer can improve slightly
    #      beyond what already exists — but never wanders into empty space.
    #
    #   3. Use the best candidates as seeds. All are real airfoils the MLP
    #      has seen — predictions are interpolation, not extrapolation.
    #      No LHS random explorers that land in empty CST space.
    #
    # Result: shapes look like real airfoils because they ARE small
    # perturbations of real airfoils.

    # ── POPULATION-ANCHORED SEARCH MODE ─────────────────────────────────────
    # Strategy:
    #   1. Filter training data: exact Re (closest), alpha ±2°, CM constraint,
    #      and CL within ±20% of target. These are real airfoils that already
    #      perform close to what you need.
    #   2. Rank by CD (target_lift) or CL/CD (best_efficiency). Take top 20.
    #   3. Derive GP search bounds from the CST min/max of all 20 candidates
    #      ± 10% exploration margin. The GP can now explore the space BETWEEN
    #      proven real airfoils — interpolating rather than extrapolating.
    #   4. Use all 20 as seeds directly. No random perturbations needed —
    #      the diversity is already in the population.
    #
    # Why ±20% CL window:
    #   Too tight (±5%) → only 5–10 candidates, GP has no diversity to explore.
    #   Too wide (±50%) → includes airfoils with very different shapes, bounds
    #   expand to cover unrelated geometries.
    #   ±20% captures airfoils that are aerodynamically similar — same regime,
    #   similar camber — giving the GP a coherent neighbourhood to search in.
    #
    # Why top 20:
    #   Enough diversity for the GP to find interpolated improvements.
    #   All verified geometry-valid before being used as seeds.
    #   GP starts with 20 real data points — builds a reliable surrogate fast.

    EXPLORE_MARGIN = 0.30   # 10% beyond population CST range per coefficient
    N_SEEDS        = 20     # top candidates to use as seeds
    CL_WINDOW      = 0.20   # ±20% of CL target

    print('\n  Setting up search space...')
    _, cst_cols = compute_cst_bounds(DATA_CSV)

    # ── Step 1: Filter training data ─────────────────────────────────────────
    print('  Filtering training data for matching conditions...')
    df_train = pd.read_csv(DATA_CSV)
    df_train = df_train[df_train['split'] == 'train'].copy()
    df_train['cl_cd'] = df_train['cl'] / df_train['cd']

    re_vals_available = df_train['re'].unique()
    closest_re_train  = re_vals_available[np.argmin(np.abs(re_vals_available - re_val))]

    nearby = df_train[
        (df_train['re'] == closest_re_train) &
        (df_train['alpha'].between(alpha_val - 2, alpha_val + 2)) &
        (df_train['cm'] >= cm_min_val)
    ].copy()
    if len(nearby) == 0:
        # Relax alpha window if nothing found
        nearby = df_train[
            (df_train['re'] == closest_re_train) &
            (df_train['cm'] >= cm_min_val)
        ].copy()

    # ── Step 2: CL filter ±20%, score ALL candidates through MLP, rank ───────
    # Key principle: training data CL was recorded at a specific Re/alpha.
    # Your query Re/alpha may differ. Score every candidate through the MLP
    # at the EXACT requested conditions before ranking — this ensures seeds
    # are genuinely good at YOUR flight condition, not just in training data.

    if _opt_mode == 'target_lift':
        cl_lo = cl_target_val * (1 - CL_WINDOW)
        cl_hi = cl_target_val * (1 + CL_WINDOW)
        candidates = nearby[nearby['cl'].between(cl_lo, cl_hi)].copy()

        # Progressive relaxation if not enough candidates
        for relax_mult in [2, 3]:
            if len(candidates) >= 5:
                break
            cl_lo = cl_target_val * (1 - CL_WINDOW * relax_mult)
            cl_hi = cl_target_val * (1 + CL_WINDOW * relax_mult)
            candidates = nearby[nearby['cl'].between(cl_lo, cl_hi)].copy()
            if len(candidates) > 0:
                print(f'  ⚑ Relaxed CL window to ±{CL_WINDOW*relax_mult*100:.0f}% '
                      f'to find {len(candidates)} candidates.')
    else:
        candidates = nearby.copy()

    if 'airfoil' in candidates.columns and len(candidates) > 0:
        candidates = candidates.drop_duplicates(subset='airfoil')

    # ── MLP re-scoring at exact requested conditions ──────────────────────────
    # Score all candidates through the MLP at the real Re/alpha you asked for.
    # Training data CL was measured at closest_re_train — which may not be
    # your requested re_val. MLP scoring corrects for this gap.
    if len(candidates) > 0:
        print(f'  Scoring {len(candidates)} candidates via MLP at exact conditions...')
        mlp_cls, mlp_cds, mlp_clcds = [], [], []
        for _, row in candidates.iterrows():
            cst_vec = [row[c] for c in cst_cols]
            p_cl, p_cd, _ = predict(_model, _scalers, cst_vec, re_val, alpha_val, _device)
            p_clcd = p_cl / p_cd if p_cd > 0 else 0.0
            mlp_cls.append(p_cl)
            mlp_cds.append(p_cd)
            mlp_clcds.append(p_clcd)
        candidates = candidates.copy()
        candidates['mlp_cl']   = mlp_cls
        candidates['mlp_cd']   = mlp_cds
        candidates['mlp_clcd'] = mlp_clcds
        if cl_target_val is not None:
            candidates['mlp_cl_dist'] = (candidates['mlp_cl'] - cl_target_val).abs()
        else:
            candidates['mlp_cl_dist'] = 0.0

        if _opt_mode == 'target_lift':
            # Filter by MLP-predicted CL within ±30% (wider to keep good shapes)
            mlp_ok = candidates[
                candidates['mlp_cl'].between(cl_target_val * 0.70,
                                             cl_target_val * 1.30)
            ].copy()
            if len(mlp_ok) >= 5:
                candidates = mlp_ok
            # Sort: best MLP CL/CD first (efficiency), then CL closeness
            candidates = candidates.sort_values(
                ['mlp_clcd', 'mlp_cl_dist'],
                ascending=[False, True]
            )
        else:
            # best_efficiency: sort purely by MLP CL/CD
            candidates = candidates.sort_values('mlp_clcd', ascending=False)

    # Take top N_SEEDS
    candidates = candidates.head(N_SEEDS)
    n_candidates = len(candidates)

    print(f'  Closest Re used : {closest_re_train:.0e}  '
          f'(requested {re_val:.0e})')
    if _opt_mode == 'target_lift':
        print(f'  CL window       : [{cl_lo:.3f}, {cl_hi:.3f}]  (±{CL_WINDOW*100:.0f}% of {cl_target_val})')
    print(f'  Candidates found: {n_candidates} (top {N_SEEDS} by '
          f'{"CD" if _opt_mode == "target_lift" else "CL/CD"})')

    if n_candidates > 0:
        has_mlp = 'mlp_cl' in candidates.columns
        best = candidates.iloc[0]
        if has_mlp:
            print(f'  Best candidate  : {best.get("airfoil","?")}  '
                  f'MLP CL={best["mlp_cl"]:.4f}  '
                  f'MLP CD={best["mlp_cd"]:.5f}  '
                  f'MLP CL/CD={best["mlp_clcd"]:.1f}')
        else:
            print(f'  Best candidate  : {best.get("airfoil","?")}  '
                  f'CL={best["cl"]:.4f}  CD={best["cd"]:.5f}  '
                  f'CL/CD={best["cl_cd"]:.1f}')
        print(f'  Seed airfoils   : (sorted by MLP CL/CD at Re={re_val:.0e} alpha={alpha_val}°)')
        for i, (_, row) in enumerate(candidates.iterrows()):
            if has_mlp:
                marker = '★' if i == 0 else ' '
                print(f'   {marker}{i+1:>2}. {row.get("airfoil","?"):<22} '
                      f'MLP CL={row["mlp_cl"]:.4f}  '
                      f'MLP CD={row["mlp_cd"]:.5f}  '
                      f'MLP CL/CD={row["mlp_clcd"]:.1f}')
            else:
                print(f'    {i+1:>2}. {row.get("airfoil","?"):<22} '
                      f'CL={row["cl"]:.4f}  CD={row["cd"]:.5f}  '
                      f'CL/CD={row["cl_cd"]:.1f}')

    # ── Step 3: Derive bounds from TOP-10 only (tighter, more coherent) ───────
    # Bounds from top-10 MLP CL/CD keep GP in the high-efficiency region.
    # Using all 20 spans completely different airfoil families — too wide.
    N_BOUNDS = min(10, n_candidates)

    if n_candidates == 0:
        print(f'  ⚠ No candidates found — falling back to global bounds.')
        bounds, _ = compute_cst_bounds(DATA_CSV)
    else:
        top_bounds_df = candidates.head(N_BOUNDS)
        cand_cst = top_bounds_df[cst_cols].values
        bounds = []
        for j in range(len(cst_cols)):
            col_vals = cand_cst[:, j]
            lo   = col_vals.min()
            hi   = col_vals.max()
            span = max(hi - lo, 0.05)
            bounds.append((float(lo - EXPLORE_MARGIN * span),
                           float(hi + EXPLORE_MARGIN * span)))

        lo_arr_diag = np.array([b[0] for b in bounds])
        hi_arr_diag = np.array([b[1] for b in bounds])
        mean_span = float((hi_arr_diag - lo_arr_diag).mean())
        print(f'  Search bounds   : top-{N_BOUNDS} MLP-ranked CST range ± {EXPLORE_MARGIN*100:.0f}%')
        print(f'    Global: [{lo_arr_diag.min():.3f}, {hi_arr_diag.max():.3f}]  '
              f'mean span={mean_span:.4f}')
        if mean_span > 0.30:
            print(f'  ⚑ Bounds are wide — GP will explore diverse geometry.')
        else:
            print(f'  ✓ Tight bounds — GP focused on high-efficiency neighbourhood.')

    space = [Real(lo, hi, name=col) for (lo, hi), col in zip(bounds, cst_cols)]

    # ── Step 4: Seeds = all top-20 candidates directly ───────────────────────
    # No perturbations needed — the diversity is in the real airfoils themselves.
    lo_arr = np.array([b[0] for b in bounds])
    hi_arr = np.array([b[1] for b in bounds])

    anchor_csts = []
    if n_candidates > 0:
        for i in range(len(candidates)):
            cst_row = candidates[cst_cols].iloc[i].tolist()
            anchor_csts.append(cst_row)

    x0 = [list(np.clip(xp, lo_arr, hi_arr)) for xp in anchor_csts]
    y0 = [objective(xp) for xp in x0]

    n_feasible_seed = sum(1 for e in _eval_log if e['feasible'])
    print(f'  Seeds           : {len(x0)} real airfoils from training data')
    print(f'  Feasible seeds  : {n_feasible_seed} / {len(x0)}')

    # ── Bayesian Optimization over population ────────────────────────────────
    # Seeds are the top-20 real airfoils evaluated above (x0/y0).
    # GP now explores the space BETWEEN them — interpolating shapes that
    # none of the 20 individually are, but that sit in the aerodynamic
    # neighbourhood defined by the population.
    print(f'\n  Running Bayesian optimization ({n_calls} iterations)...')
    print(f'  GP interpolating across {len(x0)}-airfoil population.\n')

    iteration_counter = [0]

    def callback(res):
        iteration_counter[0] += 1
        it = iteration_counter[0]
        if it % 10 == 0 or it == 1:
            n_inv = sum(1 for e in _eval_log if e.get('invalid_geometry'))
            if _opt_mode == 'target_lift':
                cl_ok_now = [e for e in _eval_log
                             if e.get('feasible') and
                             abs(e.get('cl', 999) - _cl_target) <= CL_TOLERANCE]
                best_cd_now = min([e['cd'] for e in cl_ok_now], default=999)
                cd_str = f'{best_cd_now:.5f}' if best_cd_now < 900 else 'none yet'
                print(f'  Iter {it:>4}/{n_calls}  '
                      f'Best CD={cd_str}  '
                      f'CL-feasible={len(cl_ok_now)}/{len(_eval_log)}  '
                      f'Rejected(geom)={n_inv}')
            else:
                best_clcd = max(
                    [e['cl_cd'] for e in _eval_log
                     if e.get('feasible') and not np.isnan(e.get('cl_cd', np.nan))],
                    default=0)
                n_feas = sum(1 for e in _eval_log if e.get('feasible'))
                print(f'  Iter {it:>4}/{n_calls}  '
                      f'Best CL/CD={best_clcd:.3f}  '
                      f'Feasible={n_feas}/{len(_eval_log)}  '
                      f'Rejected(geom)={n_inv}')
        # Write progress for dashboard
        try:
            n_feas = sum(1 for e in _eval_log if e['feasible'])
            best_c = min((e['cd'] for e in _eval_log
                          if e.get('feasible') and not np.isnan(e.get('cd', np.nan))),
                         default=999.0)
            with open(os.path.join(OUT_DIR, 'progress.txt'), 'w') as pf:
                pf.write(f'{it},{n_calls},{n_feas},{best_c:.6f},single\n')
        except Exception:
            pass

    gp_minimize(
        func             = objective,
        dimensions       = space,
        n_calls          = n_calls,
        n_initial_points = max(20, len(x0) + 10),
        x0               = x0,
        y0               = y0,
        acq_func         = 'EI',
        noise            = 1e-4,
        random_state     = 42,
        callback         = callback,
        verbose          = False,
    )


    # ── Extract best result ───────────────────────────────────────────────────
    print(f'\n  Optimization complete.')

    # Find best FEASIBLE solution — feasible means BOTH CL tolerance AND CM satisfied
    feasible_evals = [e for e in _eval_log
                      if e['feasible'] and not np.isnan(e.get('cl_cd', np.nan))]

    if not feasible_evals:
        # No strictly feasible result — fall back gracefully
        # For target_lift: pick the evaluation closest to CL target
        # For efficiency:  pick the best CL/CD ignoring CM (warn user)
        valid = [e for e in _eval_log if not np.isnan(e.get('cl_cd', np.nan))
                 and not np.isnan(e.get('cl', np.nan))]
        if not valid:
            # Every optimizer evaluation was rejected by geometry check.
            # Your principle: always return a real answer.
            # Fall back to the best matching airfoil from the training data
            # directly — no optimizer improvement, but a proven real airfoil
            # that satisfies the target condition. Better than nothing.
            print('\n  ⚠ All optimizer proposals rejected by geometry check.')
            print('  Falling back to best matching airfoil from training data.')
            _df_fb = pd.read_csv(DATA_CSV)
            _df_fb = _df_fb[_df_fb['split'] == 'train'].copy()
            _re_vals = _df_fb['re'].unique()
            _closest_re = _re_vals[np.argmin(np.abs(_re_vals - _re))]
            _nearby_fb = _df_fb[
                (_df_fb['re'] == _closest_re) &
                (_df_fb['alpha'].between(_alpha - 3, _alpha + 3))
            ].copy()
            _cst_cols_fb = [f'cst_u{i}' for i in range(10)] + [f'cst_l{i}' for i in range(10)]
            if _opt_mode == 'target_lift':
                _fb_pool = _nearby_fb[
                    _nearby_fb['cl'].between(_cl_target - CL_TOLERANCE * 3,
                                              _cl_target + CL_TOLERANCE * 3)
                ].copy()
                if len(_fb_pool) == 0:
                    _fb_pool = _nearby_fb.copy()
                _fb_pool['_cl_dist'] = (_fb_pool['cl'] - _cl_target).abs()
                _best_fb = _fb_pool.sort_values(['_cl_dist', 'cd']).iloc[0]
            else:
                _nearby_fb['_cl_cd'] = _nearby_fb['cl'] / _nearby_fb['cd']
                _best_fb = _nearby_fb.sort_values('_cl_cd', ascending=False).iloc[0]
            _cst_fb = _best_fb[_cst_cols_fb].tolist()
            _pred   = predict(_model, _scalers, _cst_fb, _re, _alpha, _device)
            valid = [{
                'cst_params': _cst_fb,
                'cl'     : float(_pred[0]),
                'cd'     : float(_pred[1]),
                'cm'     : float(_pred[2]),
                'cl_cd'  : float(_pred[0] / _pred[1]) if _pred[1] > 0 else 0,
                'feasible': True,
                'source' : f'training_data_fallback:{_best_fb["airfoil"]}',
            }]
            print(f'  Fallback airfoil  : {_best_fb["airfoil"]}')
            print(f'  Training data CL  : {_best_fb["cl"]:.4f}  CD={_best_fb["cd"]:.5f}')
            print(f'  MLP prediction    : CL={_pred[0]:.4f}  CD={_pred[1]:.5f}  CM={_pred[2]:.4f}')

        if _opt_mode == 'target_lift':
            # closest to CL target among all valid evals
            fallback = min(valid, key=lambda e: abs(e['cl'] - _cl_target))
            print(f'\n  ⚠ No result met both CL tolerance AND CM constraint.')
            print(f'  Returning closest result to CL target (CL={fallback["cl"]:.4f}).')
            print(f'  Suggestions:')
            print(f'    → Relax stability: --stability C or D')
            print(f'    → Run more iterations: --n_calls 250')
            print(f'    → Adjust alpha — CL={_cl_target} may need higher alpha')
            feasible_evals = [fallback]
        else:
            fallback = max(valid, key=lambda e: e['cl_cd'])
            print(f'\n  ⚠ No CM-feasible result found. Returning best CL/CD ignoring CM.')
            print(f'  Try: relax stability → --stability C or D')
            feasible_evals = [fallback]

    # ── Best result selector — double-verified ───────────────────────────────
    # Do NOT rely solely on the stored 'feasible' flag from the eval log.
    # Re-verify the CL constraint here so a stale or incorrectly-set flag
    # cannot corrupt the final result. This is the last line of defence.
    if _opt_mode == 'target_lift':
        # Re-filter: only evals genuinely within CL tolerance
        cl_strict = [e for e in feasible_evals
                     if abs(e['cl'] - _cl_target) <= CL_TOLERANCE]
        if cl_strict:
            best_eval = min(cl_strict, key=lambda e: e['cd'])
        else:
            # All "feasible" evals failed the re-check — pick closest CL to target
            best_eval = min(feasible_evals, key=lambda e: abs(e['cl'] - _cl_target))
            print(f'  ⚠ Re-verification: no eval within CL tolerance in feasible set.')
            print(f'    Returning closest CL result. Penalty scaling may need tuning.')
    else:
        best_eval = max(feasible_evals, key=lambda e: e['cl_cd'])
    best_cst   = best_eval['cst_params']
    best_cl    = best_eval['cl']
    best_cd    = best_eval['cd']
    best_cm    = best_eval['cm']
    best_clcd  = best_eval['cl_cd']
    n_feasible = len(feasible_evals)

    print(f'\n{"═"*65}')
    print(f'  BEST RESULT')
    print(f'{"═"*65}')
    print(f'  CL       : {best_cl:.4f}', end='')
    if _opt_mode == 'target_lift':
        diff = best_cl - _cl_target
        print(f'  (target {_cl_target:.2f}, diff {diff:+.4f}'
              f'  {"✓ within tolerance" if abs(diff) <= CL_TOLERANCE else "⚠ outside tolerance"})')
    else:
        print()
    print(f'  CD       : {best_cd:.6f}')
    print(f'  CM       : {best_cm:.5f}  {"✓ stable" if best_cm >= cm_min_val else "⚠ check"}')
    t_min, t_max = _get_thickness_stats(best_cst)
    print(f'  CL/CD    : {best_clcd:.3f}')
    print(f'  Thickness: {t_min:.2f}% min  —  {t_max:.2f}% max chord')
    print(f'  Feasible : {n_feasible} / {n_calls + len(x0)} evaluations')
    print(f'{"═"*65}')

    # ── Reconstruct airfoil ───────────────────────────────────────────────────
    print('\n  Reconstructing airfoil shape from CST parameters...')
    coords, x_arr, y_upper, y_lower = cst_to_coordinates(best_cst)

    airfoil_name = f'opt_Re{int(re_val)}_a{alpha_val:.0f}'
    dat_path     = os.path.join(OUT_DIR, 'best_airfoil.dat')
    save_dat_file(coords, airfoil_name, dat_path)

    # ── Save all evaluations ──────────────────────────────────────────────────
    eval_df = pd.DataFrame([
        {k: v for k, v in e.items() if k != 'cst_params'}
        for e in _eval_log
    ])
    eval_path = os.path.join(OUT_DIR, 'all_evaluations.csv')
    eval_df.to_csv(eval_path, index=False)
    print(f'  All evaluations → {eval_path}')

    # ── Plots ─────────────────────────────────────────────────────────────────
    print('\n  Generating plots...')
    plot_convergence_curve(
        _eval_log, re_val, alpha_val,
        os.path.join(OUT_DIR, 'convergence.png')
    )
    plot_airfoil(
        x_arr, y_upper, y_lower,
        best_cl, best_cd, best_cm, re_val, alpha_val,
        os.path.join(OUT_DIR, 'best_airfoil.png')
    )

    # ── Report ────────────────────────────────────────────────────────────────
    save_report(
        best_cst, best_cl, best_cd, best_cm,
        re_val, alpha_val, n_calls + len(x0),
        n_feasible, _eval_log,
        f'[{stability_key}] {stability_desc}',
        cm_min_val,
        os.path.join(OUT_DIR, 'optimization_report.txt'),
        opt_mode=opt_mode,
        cl_target=cl_target_val,
    )

    # ── SHAP explanation of the best airfoil ──────────────────────────────────
    # Runs on the single best result only — fast (seconds not minutes).
    # Answers: WHY did the model recommend this specific geometry?
    if SHAP_AVAILABLE:
        X_bg = build_shap_background(_scalers, DATA_CSV, n=50)
        explain_best_airfoil(
            _model, _scalers,
            best_cst, re_val, alpha_val,
            best_cl, best_cd, best_cm,
            X_bg, stability_desc,
            os.path.join(OUT_DIR, 'shap_explanation.png')
        )
    else:
        print('\n  ⚠ SHAP explanation skipped (shap not installed).')
        print('     Install: pip install shap --break-system-packages')

    print('\n' + '█'*65)
    print('  DONE.')
    print(f'  Best airfoil saved → {dat_path}')
    print(f'  Validate in XFOIL to confirm predictions.')
    print('█'*65 + '\n')


if __name__ == '__main__':
    main()