#!/usr/bin/python3

"""
Airfoil CST Bayesian Optimizer — Streamlit Dashboard
=====================================================
Three pages:
  1. Single Phase  — direct OR physical inputs, single flight condition
  2. Multi Phase   — direct OR physical inputs, multi-phase compromise
  3. Results       — shared output: airfoil plot, SHAP geometric map, report

Run:
    streamlit run app.py
"""

import os, sys, json, time, subprocess, threading, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import streamlit as st
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.resolve()
RESULTS_ROOT = BASE_DIR / 'optimizer_results'
RESULTS_ROOT.mkdir(exist_ok=True)

OPTIMIZER_SCRIPT = BASE_DIR / 'b2.py'
FLIGHT_PROFILE   = BASE_DIR / 'flight_profile.json'


def get_latest_run_dir() -> Path:
    """
    Return the most recent run directory.
    Tries the 'latest' symlink first (Linux/Mac).
    Falls back to 'latest.txt' (Windows).
    Falls back to scanning for the most recently modified subfolder.
    """
    # Symlink (Linux/Mac)
    sym = RESULTS_ROOT / 'latest'
    if sym.is_symlink() and sym.resolve().exists():
        return sym.resolve()

    # Windows text fallback
    txt = RESULTS_ROOT / 'latest.txt'
    if txt.exists():
        p = Path(txt.read_text().strip())
        if p.exists():
            return p

    # Last resort — find newest subfolder
    subdirs = [d for d in RESULTS_ROOT.iterdir()
               if d.is_dir() and d.name.startswith('run_')]
    if subdirs:
        return max(subdirs, key=lambda d: d.stat().st_mtime)

    return RESULTS_ROOT   # absolute fallback — empty state


def get_all_runs() -> list:
    """Return all run dirs sorted newest-first."""
    subdirs = [d for d in RESULTS_ROOT.iterdir()
               if d.is_dir() and d.name.startswith('run_')]
    return sorted(subdirs, key=lambda d: d.stat().st_mtime, reverse=True)


# OUT_DIR is resolved dynamically per page render so new runs appear immediately
OUT_DIR = get_latest_run_dir()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='AirfoilOpt',
    page_icon='✈',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Design system ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

/* ══ DARK MODE (default) ══════════════════════════════════════════════ */
.stApp { background: #0a0c10; color: #e8eaf0; }

[data-testid="stSidebar"] {
    background: #0f1219 !important;
    border-right: 1px solid #1e2330;
}

h1 { font-size:2rem !important; font-weight:800 !important;
     letter-spacing:-0.03em; color:#ffffff; }
h2 { font-size:1.15rem !important; font-weight:600 !important;
     color:#c8cfe0; border-bottom:1px solid #1e2330; padding-bottom:6px; }

.metric-card {
    background: #121620; border: 1px solid #1e2330;
    border-radius: 8px; padding: 16px 20px; margin-bottom: 10px;
}
.metric-label {
    font-family: 'Space Mono', monospace; font-size: 0.64rem;
    color: #3a4a5a; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 5px;
}
.metric-value {
    font-family: 'Space Mono', monospace; font-size: 1.3rem;
    font-weight: 700; color: #4fc3f7;
}
.metric-sub {
    font-family: 'Space Mono', monospace; font-size: 0.68rem;
    color: #8892a4; margin-top: 3px;
}
.computed-badge {
    display: inline-block; background: #0d1a2a;
    border: 1px solid #1e3a5a; border-radius: 4px;
    padding: 4px 10px; font-family: 'Space Mono', monospace;
    font-size: 0.68rem; color: #64b5f6; margin: 3px 4px 3px 0;
}
.info-box {
    background: #0d1a2a; border: 1px solid #1e3a5a; border-radius: 6px;
    padding: 12px 16px; font-family: 'Space Mono', monospace;
    font-size: 0.7rem; color: #64b5f6; margin: 8px 0;
}
.warn-box {
    background: #1a1500; border: 1px solid #3d3000; border-radius: 6px;
    padding: 12px 16px; font-family: 'Space Mono', monospace;
    font-size: 0.7rem; color: #ffd54f; margin: 8px 0;
}
.ok-box {
    background: #0d1a12; border: 1px solid #1a3a22; border-radius: 6px;
    padding: 12px 16px; font-family: 'Space Mono', monospace;
    font-size: 0.7rem; color: #81c784; margin: 8px 0;
}
.phase-wrap {
    background: #0f1520; border: 1px solid #1e2a3a;
    border-radius: 10px; padding: 20px 22px; margin-bottom: 14px;
}
.progress-log {
    background: #080c12; border: 1px solid #1e2330; border-radius: 6px;
    padding: 14px 18px; font-family: 'Space Mono', monospace;
    font-size: 0.72rem; color: #4fc3f7; line-height: 1.9;
}

/* ══ LIGHT MODE — fires when OS/Streamlit config is light ═════════════ */
@media (prefers-color-scheme: light) {
    .stApp { background: #f5f7fa !important; color: #1a202c !important; }

    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] * { color: #1a202c !important; }

    h1 { color: #1a202c !important; }
    h2 { color: #2d3748 !important; border-bottom-color: #e2e8f0 !important; }

    .metric-card {
        background: #ffffff !important;
        border-color: #e2e8f0 !important;
    }
    .metric-label { color: #718096 !important; }
    .metric-value { color: #1565c0 !important; }
    .metric-sub   { color: #4a5568 !important; }

    .computed-badge {
        background: #ebf8ff !important;
        border-color: #bee3f8 !important;
        color: #2b6cb0 !important;
    }
    .info-box {
        background: #ebf8ff !important;
        border-color: #bee3f8 !important;
        color: #2b6cb0 !important;
    }
    .warn-box {
        background: #fffff0 !important;
        border-color: #f6e05e !important;
        color: #744210 !important;
    }
    .ok-box {
        background: #f0fff4 !important;
        border-color: #9ae6b4 !important;
        color: #276749 !important;
    }
    .phase-wrap {
        background: #ffffff !important;
        border-color: #e2e8f0 !important;
    }
    .progress-log {
        background: #f7fafc !important;
        border-color: #e2e8f0 !important;
        color: #1a202c !important;
    }
}

/* ── Mode selector cards ── */
.mode-card {
    background: #111520;
    border: 2px solid #1e2a3a;
    border-radius: 12px;
    padding: 26px 28px;
    margin-bottom: 4px;
    transition: all 0.2s ease;
    position: relative;
    min-height: 170px;
}
.mode-card.active {
    border-color: #4fc3f7;
    background: #0c1828;
    box-shadow: 0 0 30px rgba(79,195,247,0.10);
}
.mode-card-icon { font-size: 2rem; margin-bottom: 12px; }
.mode-card-title {
    font-size: 1.05rem; font-weight: 800;
    color: #fff; margin-bottom: 8px; letter-spacing: -0.01em;
}
.mode-card.active .mode-card-title { color: #4fc3f7; }
.mode-card-desc {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem; color: #4a5a6a; line-height: 1.8;
}
.mode-card-badge {
    position: absolute; top: 14px; right: 14px;
    background: #4fc3f7; color: #000;
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem; font-weight: 700;
    padding: 3px 9px; border-radius: 20px; letter-spacing: 0.08em;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    color: #fff !important; border: none !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important; font-weight: 700 !important;
    letter-spacing: 0.06em !important; padding: 10px 24px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1976d2, #1565c0) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(21,101,192,0.4) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.06em; color: #4a5568;
}
.stTabs [aria-selected="true"] {
    color: #4fc3f7 !important; border-bottom-color: #4fc3f7 !important;
}

[data-testid="stSlider"] label,
.stSelectbox label, .stRadio label, .stNumberInput label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important; color: #8892a4 !important;
}

hr { border-color: #1e2330 !important; margin: 20px 0 !important; }

.empty-state { text-align:center; padding:60px 20px; }
.empty-state-icon { font-size:3rem; margin-bottom:12px; }
.empty-state-text {
    font-family:'Space Mono',monospace; font-size:0.78rem; color:#3a4255; line-height:2;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS & UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

STABILITY_MODES = {
    'A': ('CM > -0.01', 'Tailless / flying wing'),
    'B': ('CM > -0.05', 'Conventional aircraft'),
    'C': ('CM > -0.10', 'Large tail / relaxed stability'),
    'D': ('CM > -∞',    'No CM constraint'),
}
STABILITY_CM  = {'A': -0.01, 'B': -0.05, 'C': -0.10, 'D': -999.0}
PHASE_COLOURS = ['#4fc3f7', '#81c784', '#ffb74d', '#e57373', '#ce93d8']
TRAINING_RES  = [50000, 100000, 200000, 300000, 500000, 700000, 1000000]
N_CST         = 10

# CST index → chord fraction (start, end)
CST_CHORD_MAP = {
    0: (0.00, 0.10), 1: (0.10, 0.22), 2: (0.22, 0.35),
    3: (0.35, 0.50), 4: (0.50, 0.62), 5: (0.62, 0.73),
    6: (0.73, 0.82), 7: (0.82, 0.90), 8: (0.90, 0.95),
    9: (0.95, 1.00),
}

# Aerodynamic feature names — maps raw CST feature names to plain-English descriptions
# Used in SHAP bar chart and geometry map annotations
AERO_FEATURE_NAMES = {
    # Upper surface (suction surface)
    'cst_u0': 'Suction surface  0–10%c  (leading edge nose)',
    'cst_u1': 'Suction surface  10–22%c  (peak suction zone)',
    'cst_u2': 'Suction surface  22–35%c  (forward upper)',
    'cst_u3': 'Suction surface  35–50%c  (mid-chord upper)',
    'cst_u4': 'Suction surface  50–62%c  (transition region)',
    'cst_u5': 'Suction surface  62–73%c  (aft upper)',
    'cst_u6': 'Suction surface  73–82%c  (pressure recovery)',
    'cst_u7': 'Suction surface  82–90%c  (near trailing edge)',
    'cst_u8': 'Suction surface  90–95%c  (trailing edge upper)',
    'cst_u9': 'Suction surface  95–100%c  (trailing edge tip)',
    # Lower surface (pressure surface)
    'cst_l0': 'Pressure surface  0–10%c  (leading edge nose)',
    'cst_l1': 'Pressure surface  10–22%c  (forward lower)',
    'cst_l2': 'Pressure surface  22–35%c  (mid-lower forward)',
    'cst_l3': 'Pressure surface  35–50%c  (mid-chord lower)',
    'cst_l4': 'Pressure surface  50–62%c  (aft lower)',
    'cst_l5': 'Pressure surface  62–73%c  (rear lower)',
    'cst_l6': 'Pressure surface  73–82%c  (pressure recovery)',
    'cst_l7': 'Pressure surface  82–90%c  (near trailing edge)',
    'cst_l8': 'Pressure surface  90–95%c  (trailing edge lower)',
    'cst_l9': 'Pressure surface  95–100%c  (trailing edge tip)',
    # Operating conditions
    're':        'Reynolds Number  (flight speed × chord / viscosity)',
    'alpha':     'Angle of Attack  α  (nose-up pitch angle)',
    'xtr_top':   'Transition location  upper surface  (laminar→turbulent)',
    'xtr_bot':   'Transition location  lower surface  (laminar→turbulent)',
}


def isa_atmosphere(alt_m):
    T0, P0, L, R, g = 288.15, 101325.0, 0.0065, 287.05, 9.80665
    mu0 = 1.716e-5
    if alt_m <= 11000:
        T = T0 - L * alt_m
        P = P0 * (T / T0) ** (g / (L * R))
    else:
        T = 216.65
        P = 22632.1 * np.exp(-g * (alt_m - 11000) / (R * T))
    rho = P / (R * T)
    mu  = mu0 * (T / T0) ** 1.5 * (T0 + 110.4) / (T + 110.4)
    return rho, mu


def compute_re(v, c, alt):
    rho, mu = isa_atmosphere(alt)
    return rho * v * c / mu


def compute_cl_phys(weight, area, v, alt, unit):
    W = weight * 9.80665 if unit == 'kg' else weight
    rho, _ = isa_atmosphere(alt)
    return max(0.0, min(2.5, 2 * W / (rho * v**2 * area)))


def snap_re(re):
    return float(min(TRAINING_RES, key=lambda x: abs(x - re)))


def read_progress():
    try:
        # Always read from the latest run dir — not the stale module-level OUT_DIR
        live_dir = get_latest_run_dir()
        parts = (live_dir / 'progress.txt').read_text().strip().split(',')
        return int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3]), parts[4]
    except Exception:
        return 0, 1, 0, 0.0, 'single'


def results_exist():
    return (OUT_DIR / 'best_airfoil.dat').exists()


def read_report():
    rp = OUT_DIR / 'optimization_report.txt'
    return rp.read_text() if rp.exists() else ''


def run_subprocess(cmd):
    # Always resolve latest dir at call time — a new run dir may have been
    # created since the module was loaded.
    live_dir = get_latest_run_dir()
    pf = live_dir / 'progress.txt'
    if pf.exists():
        pf.unlink()
    proc = subprocess.Popen([sys.executable] + cmd,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        lines.append(line)
    proc.wait()
    return proc.returncode, ''.join(lines)


# ── CST reconstruction ────────────────────────────────────────────────────────

def bernstein(x, n, k):
    from math import comb
    return comb(n, k) * (x**k) * ((1-x)**(n-k))


def cst_surface(x, weights):
    n   = len(weights) - 1
    cls = x**0.5 * (1 - x)
    return cls * sum(weights[k] * bernstein(x, n, k) for k in range(n+1))


def cst_to_xy(cst, n=200):
    xs  = np.linspace(1e-6, 1, n)
    wu, wl = cst[:N_CST], cst[N_CST:2*N_CST]
    yu  = np.array([cst_surface(xi, wu) for xi in xs])
    yl  = np.array([cst_surface(xi, wl) for xi in xs])
    return xs, yu, yl


def read_dat_xy(dat_path):
    """
    Read airfoil coordinates directly from best_airfoil.dat.
    Returns (x, y_upper, y_lower) on a uniform x grid.
    This is always consistent with the displayed airfoil shape.
    """
    try:
        lines = Path(dat_path).read_text().splitlines()
        pts = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
        if len(pts) < 10:
            return None, None, None

        pts = np.array(pts)
        # Find split between upper and lower surface
        # Dat files go: TE -> upper LE -> TE -> lower (or vice versa)
        # Split at the minimum x point (leading edge)
        le_idx = int(np.argmin(pts[:, 0]))
        upper = pts[:le_idx + 1][::-1]   # LE to TE upper
        lower = pts[le_idx:]              # LE to TE lower

        if len(upper) < 3 or len(lower) < 3:
            return None, None, None

        # Interpolate both surfaces onto uniform x grid
        x_grid = np.linspace(0.001, 0.999, 200)
        from scipy.interpolate import interp1d
        try:
            fu = interp1d(upper[:, 0], upper[:, 1],
                          kind='linear', bounds_error=False, fill_value='extrapolate')
            fl = interp1d(lower[:, 0], lower[:, 1],
                          kind='linear', bounds_error=False, fill_value='extrapolate')
            return x_grid, fu(x_grid), fl(x_grid)
        except Exception:
            return None, None, None
    except Exception:
        return None, None, None


# ── SHAP geometric map ────────────────────────────────────────────────────────

def load_shap_values(run_dir=None):
    d  = Path(run_dir) if run_dir else OUT_DIR
    sv = d / 'shap_values.json'
    if sv.exists():
        try:
            return json.loads(sv.read_text())
        except Exception:
            return None
    return None


def make_shap_airfoil_figure(x, yu, yl, shap_data):
    """
    3 rows (CL, CD, CM), 1 column.
    Single shared colorbar at top. Generous spacing between panels.
    """
    REGION_NAMES = {
        0: 'Leading edge (nose)',    1: 'Near leading edge',
        2: 'Forward upper surface',  3: 'Forward lower surface',
        4: 'Mid-chord upper',        5: 'Mid-chord lower',
        6: 'Aft upper surface',      7: 'Aft lower surface',
        8: 'Near trailing edge',     9: 'Trailing edge',
    }

    x, yu, yl = np.array(x), np.array(yu), np.array(yl)
    feat_names = shap_data.get('feature_names', [])
    shap_cl    = np.array(shap_data.get('shap_cl', []))
    shap_cd    = np.array(shap_data.get('shap_cd', []))
    shap_cm    = np.array(shap_data.get('shap_cm', []))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'shap_geo',
        ['#b71c1c','#ef5350','#ffcdd2','#f5f5f5','#bbdefb','#1976d2','#0d47a1'],
        N=256,
    )
    all_shap = np.concatenate([shap_cl, shap_cd, shap_cm])
    abs_max  = max(float(np.abs(all_shap).max()), 1e-8) if len(all_shap) > 0 else 1.0
    norm     = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

    titles = [
        'CL — Lift Coefficient  (blue = region increases lift)',
        'CD — Drag Coefficient  (blue = region increases drag)',
        'CM — Pitching Moment  (blue = region increases nose-down moment)',
    ]
    arrays = [shap_cl, shap_cd, shap_cm]

    y_lo   = float(yl.min())
    y_hi   = float(yu.max())
    y_span = y_hi - y_lo
    pad    = y_span * 0.06

    # Tall figure — each airfoil panel gets plenty of vertical room
    fig = plt.figure(figsize=(16, 15))
    fig.patch.set_facecolor('#ffffff')

    # Title at very top
    fig.text(0.5, 0.985,
             'SHAP Geometry Map — Which airfoil regions drive each aerodynamic output',
             ha='center', va='top', fontsize=12, fontweight='bold',
             color='#222222', fontfamily='monospace')

    # Single shared colorbar just below title
    cbar_ax = fig.add_axes([0.12, 0.967, 0.76, 0.009])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=8, colors='#444444')
    cb.outline.set_edgecolor('#cccccc')
    cb.set_label(
        'Red ← decreases output          SHAP value          increases output → Blue',
        color='#444444', fontfamily='monospace', fontsize=8.5
    )

    # Each panel occupies a fixed vertical slice with plenty of room
    # Panel heights: each airfoil = 0.22, gap between panels = 0.08
    panel_h   = 0.14
    gap       = 0.14
    top_start = 0.87   # first panel top

    for row, (title, shap_arr) in enumerate(zip(titles, arrays)):
        bottom = top_start - row * (panel_h + gap) - panel_h
        ax = fig.add_axes([0.08, bottom, 0.86, panel_h])
        ax.set_facecolor('#ffffff')

        shap_dict = {name: float(shap_arr[j])
                     for j, name in enumerate(feat_names) if j < len(shap_arr)}

        for seg, (xs, xe) in CST_CHORD_MAP.items():
            mask = (x >= xs) & (x <= xe)
            if mask.sum() < 2:
                continue
            sv_u = shap_dict.get(f'cst_u{seg}', shap_dict.get(f'Upper CST {seg}', 0.0))
            sv_l = shap_dict.get(f'cst_l{seg}', shap_dict.get(f'Lower CST {seg}', 0.0))
            ax.fill_between(x[mask], yu[mask], 0, color=cmap(norm(sv_u)), alpha=0.75, linewidth=0)
            ax.fill_between(x[mask], yl[mask], 0, color=cmap(norm(sv_l)), alpha=0.75, linewidth=0)

        ax.fill_between(np.concatenate([x[::-1], x[1:]]),
                        np.concatenate([yu[::-1], yl[1:]]),
                        alpha=0.06, color='steelblue')
        ax.plot(x, yu, color='steelblue', lw=2.5, label='Upper surface (suction)')
        ax.plot(x, yl, color='coral',     lw=2.5, label='Lower surface (pressure)')

        ax.axhline(0, color='#cccccc', lw=0.6, ls='--')
        for loc in [0.25, 0.5, 0.75]:
            ax.axvline(loc, color='#eeeeee', lw=0.8, ls=':')
            ax.text(loc, y_lo - 0.003, f'{int(loc*100)}%c',
                    ha='center', va='top', color='#999999',
                    fontfamily='monospace', fontsize=8)

        ax.set_xlim(-0.01, 1.02)
        ax.set_ylim(y_lo - pad, y_hi + pad)
        ax.set_xlabel('x/c', fontsize=9, color='#555555', fontfamily='monospace')
        ax.set_ylabel('y/c', fontsize=9, color='#555555', fontfamily='monospace')
        ax.tick_params(labelsize=8, colors='#666666')
        for sp in ax.spines.values():
            sp.set_edgecolor('#dddddd')
        ax.grid(True, alpha=0.15)
        ax.set_title(title, fontsize=10, fontweight='bold',
                     color='#222222', fontfamily='monospace', pad=8)
        if row == 0:
            ax.legend(fontsize=8.5, loc='upper right',
                      framealpha=0.9, edgecolor='#dddddd')

        # Contributor table — placed in figure coordinates below the axes
        tbl_top = bottom - 0.05   # just below the axes bottom
        seg_list = []
        for seg in range(N_CST):
            sv_u = shap_dict.get(f'cst_u{seg}', shap_dict.get(f'Upper CST {seg}', 0.0))
            sv_l = shap_dict.get(f'cst_l{seg}', shap_dict.get(f'Lower CST {seg}', 0.0))
            seg_list.append((abs(sv_u + sv_l), sv_u + sv_l, seg))
        seg_list.sort(reverse=True)

        fig.text(0.08, tbl_top,
                 '  Rank   SHAP value   Region                    Chord range   Effect',
                 ha='left', va='top', fontsize=8, color='#444444',
                 fontfamily='monospace', fontweight='bold')
        fig.text(0.08, tbl_top - 0.006, '-' * 90,
                 ha='left', va='top', fontsize=6, color='#cccccc',
                 fontfamily='monospace')

        for rank, (_, total, seg) in enumerate(seg_list[:3]):
            col    = '#1565c0' if total >= 0 else '#c62828'
            sign   = '+' if total >= 0 else ''
            pct    = f'{CST_CHORD_MAP[seg][0]*100:.0f}-{CST_CHORD_MAP[seg][1]*100:.0f}%c'
            name   = REGION_NAMES.get(seg, f'Seg {seg}')
            effect = 'increases' if total >= 0 else 'decreases'
            line   = f'  #{rank+1}      {sign}{total:.4f}     {name:<26}  {pct:<12}  {effect}'
            fig.text(0.08, tbl_top - 0.013 - rank * 0.012, line,
                     ha='left', va='top', fontsize=8, color=col,
                     fontfamily='monospace')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    buf.seek(0)
    return buf

# ── Input mode selector ───────────────────────────────────────────────────────

def input_mode_selector(key):
    if f'mode_{key}' not in st.session_state:
        st.session_state[f'mode_{key}'] = 'direct'

    m = st.session_state[f'mode_{key}']

    cards = [
        ('direct',   '🎛️', 'Direct Technical Input',
         'Enter Reynolds number, angle of attack, and CL target directly.<br><br>'
         'For: researchers and engineers who already know their aerodynamic values.'),
        ('physical', '✈️', 'Physical Flight Calculator',
         'Enter airspeed, altitude, aircraft weight and wing geometry.<br><br>'
         'Reynolds number and CL target are computed automatically for you.'),
    ]

    # Use double quotes for ALL HTML attributes — single quotes inside f-strings
    # cause Streamlit to escape the HTML and render it as raw text
    card_divs = []
    for mode_id, icon, title, desc in cards:
        is_active  = m == mode_id
        border     = '#4fc3f7' if is_active else '#1e2a3a'
        bg         = '#0c1828' if is_active else '#111520'
        title_col  = '#4fc3f7' if is_active else '#ffffff'
        badge_html = (
            '<span style="position:absolute;top:12px;right:12px;'
            'background:#4fc3f7;color:#000;font-family:monospace;'
            'font-size:0.56rem;font-weight:700;padding:2px 8px;'
            'border-radius:20px;letter-spacing:0.08em;">ACTIVE</span>'
            if is_active else ''
        )
        card_divs.append(
            f'<div style="flex:1;position:relative;background:{bg};'
            f'border:2px solid {border};border-radius:12px;'
            f'padding:24px 24px 20px 24px;min-height:160px;">'
            f'{badge_html}'
            f'<div style="font-size:1.8rem;margin-bottom:10px;">{icon}</div>'
            f'<div style="font-size:1rem;font-weight:800;color:{title_col};'
            f'margin-bottom:8px;letter-spacing:-0.01em;">{title}</div>'
            f'<div style="font-family:Space Mono,monospace;font-size:0.67rem;'
            f'color:#6a7a8a;line-height:1.8;">{desc}</div>'
            f'</div>'
        )

    full_html = (
        '<div style="display:flex;gap:16px;margin-bottom:12px;">'
        + ''.join(card_divs)
        + '</div>'
    )
    st.markdown(full_html, unsafe_allow_html=True)

    # Buttons below the cards
    col1, col2 = st.columns(2, gap='large')
    for col, mode_id, title in [
        (col1, 'direct',   'Direct Technical Input'),
        (col2, 'physical', 'Physical Flight Calculator'),
    ]:
        with col:
            label = '✓ Using this mode' if m == mode_id else f'Select {title}'
            if st.button(label, key=f'btn_{mode_id}_{key}', use_container_width=True):
                st.session_state[f'mode_{key}'] = mode_id
                st.rerun()

    st.markdown('<hr>', unsafe_allow_html=True)
    return st.session_state[f'mode_{key}']


# ── Progress runner ───────────────────────────────────────────────────────────

def run_with_progress(cmd, metric='Best CL/CD'):
    st.markdown("## Optimization Running")
    pbar   = st.progress(0)
    status = st.empty()
    start  = time.time()
    res    = {'code': None, 'log': ''}

    def _run():
        c, l = run_subprocess(cmd)
        res['code'] = c
        res['log']  = l

    t = threading.Thread(target=_run)
    t.start()

    while t.is_alive():
        it, total, feas, best_m, _ = read_progress()
        elapsed = time.time() - start
        pbar.progress(min(int(it / max(total, 1) * 100), 100))
        best_s = f'{best_m:.5f}' if best_m < 900 else '—'
        status.markdown(f"""
        <div class='progress-log'>
            ▶ &nbsp; Iteration &nbsp;<b style='color:#fff'>{it}</b> / {total}
            &emsp;|&emsp; Feasible &nbsp;<b style='color:#81c784'>{feas}</b>
            &emsp;|&emsp; {metric} &nbsp;<b style='color:#4fc3f7'>{best_s}</b>
            &emsp;|&emsp; Elapsed &nbsp;<b style='color:#ffb74d'>{elapsed:.0f}s</b>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(1.5)

    t.join()
    pbar.progress(100)
    return res


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='padding:20px 0 24px 0;'>
        <div style='font-family:Syne,sans-serif; font-size:1.4rem;
                    font-weight:800; color:#fff; letter-spacing:-0.02em;'>
            ✈ AirfoilOpt
        </div>
        <div style='font-family:Space Mono,monospace; font-size:0.6rem;
                    color:#8892a4; margin-top:4px; letter-spacing:0.06em;'>
            CST BAYESIAN OPTIMIZER
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio('Navigation', ['⚙  Single Phase', '✈  Multi Phase', '📊  Results'],
                    label_visibility='collapsed')

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Space Mono,monospace; font-size:0.62rem;
                color:#8892a4; line-height:2.1;'>
        MODEL ACCURACY (v2 · Adj-1)<br>
        <span style='color:#4fc3f7'>CL</span> R² = 0.9592
        &nbsp; <span style='color:#4fc3f7'>CD</span> R² = 0.9468<br>
        <span style='color:#4fc3f7'>CM</span> R² = 0.9058<br><br>
        TRAINING DATA<br>
        <span style='color:#4fc3f7'>556</span> airfoil geometries<br>
        <span style='color:#4fc3f7'>93,114</span> XFOIL evaluations<br>
        <span style='color:#4fc3f7'>24</span> input features
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Space Mono,monospace; font-size:0.58rem; color:#8892a4;'>
        ⚠ Validate results in XFOIL<br>before use in design.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — SINGLE PHASE
# ══════════════════════════════════════════════════════════════════════════════

if page == '⚙  Single Phase':

    st.markdown("# Single-Phase Optimization")
    st.markdown("""
    <div style='font-family:Space Mono,monospace; font-size:0.73rem;
                color:#4a5568; margin-bottom:18px;'>
        Find the optimal CST airfoil for one flight condition.<br>
        Select how you want to provide your inputs — no confusion, no guessing.
    </div>
    """, unsafe_allow_html=True)

    mode = input_mode_selector('sp')

    left, right = st.columns([1.1, 0.9], gap='large')

    with left:
        st.markdown("## Flight Condition")

        if mode == 'direct':
            re_col1, re_col2 = st.columns([2, 1])
            with re_col1:
                re_input = st.number_input(
                    'Reynolds Number',
                    min_value=50000, max_value=1000000,
                    value=500000, step=50000,
                    help='Type any value or use arrows. Will snap to nearest trained Re.'
                )
            with re_col2:
                re_val = snap_re(re_input)
                snapped_direct = abs(re_input - re_val) / re_input > 0.10
                st.markdown(f"""
                <div style='margin-top:28px;'>
                    <span class='computed-badge'>→ {re_val:,}
                    {"⚠ snapped" if snapped_direct else "✓"}</span>
                </div>
                """, unsafe_allow_html=True)

            alpha_col1, alpha_col2 = st.columns([2, 1])
            with alpha_col1:
                alpha_val = st.slider('Angle of Attack  α (°)', -5.0, 15.0, 5.0, 0.5)
            with alpha_col2:
                alpha_val = st.number_input('α precise', -5.0, 15.0,
                                             float(alpha_val), 0.1,
                                             label_visibility='visible',
                                             key='alpha_precise_sp')
            cl_phys = None

        else:
            c1, c2 = st.columns(2)
            with c1:
                chord_sp  = st.number_input('Wing Chord  c (m)', 0.05, 10.0, 1.2, 0.05)
                airspd_sp = st.number_input('Airspeed  V (m/s)', 5.0, 400.0, 100.0, 5.0)
            with c2:
                altit_sp  = st.number_input('Altitude  (m)', 0.0, 15000.0, 3000.0, 100.0)
                alpha_val = st.number_input('Angle of Attack  α (°)', -5.0, 15.0, 5.0, 0.5)

            wu_sp = st.selectbox('Weight unit', ['kg', 'N'], key='wu_sp')
            wt_sp = st.number_input(f'Aircraft weight ({wu_sp})',
                                     1.0, 500000.0, 5000.0, 100.0)
            area_sp = st.number_input('Wing area S (m²)', 0.1, 500.0, 15.0, 0.5)

            rho_sp, mu_sp = isa_atmosphere(altit_sp)
            re_raw_sp     = rho_sp * airspd_sp * chord_sp / mu_sp
            re_val        = snap_re(re_raw_sp)
            cl_phys       = compute_cl_phys(wt_sp, area_sp, airspd_sp, altit_sp, wu_sp)

            snapped = abs(re_raw_sp - re_val) / re_raw_sp > 0.12
            st.markdown(f"""
            <div style='margin-top:10px;'>
                <span class='computed-badge'>Re = {re_val:,.0f}
                    {"⚠ snapped" if snapped else "✓"}</span>
                <span class='computed-badge'>CL implied = {cl_phys:.3f}</span>
                <span class='computed-badge'>ρ = {rho_sp:.4f} kg/m³</span>
            </div>
            """, unsafe_allow_html=True)
            if snapped:
                st.markdown(f"""
                <div class='warn-box'>
                    Computed Re = {re_raw_sp:,.0f} — snapped to nearest
                    training value {re_val:,.0f}.
                </div>
                """, unsafe_allow_html=True)

        st.markdown("## Stability Constraint")
        stab = st.radio('Stability Mode', list(STABILITY_MODES.keys()), index=1,
                         format_func=lambda k:
                         f'[{k}]  {STABILITY_MODES[k][1]}  —  {STABILITY_MODES[k][0]}')

        st.markdown("## Optimization Objective")
        obj = st.radio('Objective', [
            '🚀  Best Efficiency — maximise CL/CD',
            '🎯  Target Lift — minimise CD at a specific CL',
        ])
        cl_target = None
        if 'Target' in obj:
            default_cl = round(min(2.0, max(-0.5, cl_phys)), 2) if cl_phys else 0.8
            cl_target  = st.slider('CL Target', -0.5, 2.0, default_cl, 0.05)

    with right:
        st.markdown("## Run Configuration")
        n_sp = st.slider('Iterations', 50, 300, 150, 10)

        obj_str = ('Maximise CL/CD' if 'Best' in obj
                   else f'Minimise CD  at  CL = {cl_target}')
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Reynolds Number</div>
            <div class='metric-value'>{re_val:,}</div>
            <div class='metric-sub'>α = {alpha_val}°</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Stability</div>
            <div class='metric-value' style='font-size:0.95rem;'>
                [{stab}] {STABILITY_MODES[stab][1]}
            </div>
            <div class='metric-sub'>{STABILITY_MODES[stab][0]}</div>
        </div>
        <div class='metric-card'>
            <div class='metric-label'>Objective</div>
            <div class='metric-value' style='font-size:0.85rem;'>{obj_str}</div>
            <div class='metric-sub'>{n_sp} iterations</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        run_sp = st.button('▶  RUN SINGLE-PHASE OPTIMIZER', use_container_width=True)

    if run_sp:
        cmd = [str(OPTIMIZER_SCRIPT),
               '--re', str(re_val), '--alpha', str(alpha_val),
               '--n_calls', str(n_sp), '--stability', stab]
        if cl_target is not None:
            cmd += ['--cl_target', str(cl_target)]
        st.markdown('<hr>', unsafe_allow_html=True)
        res = run_with_progress(cmd, 'Best CL/CD')
        if res['code'] == 0:
            st.success('✓ Complete — navigate to 📊 Results')
        else:
            st.error('Optimizer error')
            st.code(res['log'][-3000:])


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — MULTI PHASE
# ══════════════════════════════════════════════════════════════════════════════

elif page == '✈  Multi Phase':

    st.markdown("# Multi-Phase Optimization")
    st.markdown("""
    <div style='font-family:Space Mono,monospace; font-size:0.73rem;
                color:#4a5568; margin-bottom:18px;'>
        Find ONE airfoil that performs well across all flight phases simultaneously.<br>
        Select how you want to define each phase — then configure below.
    </div>
    """, unsafe_allow_html=True)

    mode_mp = input_mode_selector('mp')

    if mode_mp == 'physical':
        st.markdown("## Shared Aircraft Parameters")
        st.markdown("""
        <div class='info-box'>
            These are shared across all phases. Airspeed and altitude per phase
            then determine that phase's Re and CL target automatically.
        </div>
        """, unsafe_allow_html=True)
        a1, a2, a3, a4 = st.columns(4)
        with a1: area_mp  = st.number_input('Wing Area S (m²)', 0.1, 500.0, 15.0, 0.5)
        with a2: chord_mp = st.number_input('Mean Chord c (m)', 0.05, 10.0, 1.2, 0.05)
        with a3: wu_mp    = st.selectbox('Weight unit', ['kg', 'N'], key='wu_mp')
        with a4: wt_mp    = st.number_input(f'Weight ({wu_mp})', 1.0, 500000.0, 5000.0, 100.0)
        st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown("## Flight Phases")

    info_text = (
        'Enter airspeed and altitude — Re and CL compute automatically.<br>'
        if mode_mp == 'physical' else
        'Enter Re, alpha, and CL target directly for each phase.<br>'
    )
    st.markdown(f"""
    <div class='info-box'>
        {info_text}
        <b>CD Weight</b>: how critical is drag at this phase? &nbsp; 0 = ignore, 1 = everything.<br>
        <b>Phase Weight</b>: how much does this phase contribute to the overall score?
    </div>
    """, unsafe_allow_html=True)

    if 'phases_mp' not in st.session_state:
        st.session_state.phases_mp = [
            {'name':'Climb',   'airspeed':60.0,  'altitude':1500.0,
             're':500000,  'alpha':8.0, 'cl_target':1.2,
             'stability':'B','cd_weight':0.3,'phase_weight':0.33},
            {'name':'Cruise',  'airspeed':100.0, 'altitude':3000.0,
             're':1000000, 'alpha':5.0, 'cl_target':0.8,
             'stability':'B','cd_weight':1.0,'phase_weight':0.34},
            {'name':'Descent', 'airspeed':70.0,  'altitude':500.0,
             're':300000,  'alpha':3.0, 'cl_target':0.6,
             'stability':'C','cd_weight':0.1,'phase_weight':0.33},
        ]

    phases_out = []

    for i, ph in enumerate(st.session_state.phases_mp):
        col = PHASE_COLOURS[i % len(PHASE_COLOURS)]
        st.markdown(f"""
        <div class='phase-wrap' style='border-left:3px solid {col};'>
            <div style='font-size:0.7rem; font-weight:700; letter-spacing:0.12em;
                        text-transform:uppercase; color:{col}; margin-bottom:14px;'>
                Phase {i+1}
            </div>
        """, unsafe_allow_html=True)

        r = st.columns([1.2, 1, 1, 1.4, 0.8, 0.8])
        with r[0]: name = st.text_input('Name', ph['name'], key=f'mp_n_{i}')
        with r[4]: cd_w = st.slider('CD weight', 0.0, 1.0, ph['cd_weight'], 0.1, key=f'mp_cd_{i}')
        with r[5]: ph_w = st.number_input('Phase wt', 0.01, 1.0, ph['phase_weight'], 0.01, key=f'mp_pw_{i}')

        if mode_mp == 'physical':
            with r[1]: airspd = st.number_input('Airspeed (m/s)', 5.0, 400.0, ph['airspeed'], 5.0, key=f'mp_v_{i}')
            with r[2]: altit  = st.number_input('Altitude (m)', 0.0, 15000.0, ph['altitude'], 100.0, key=f'mp_a_{i}')
            with r[3]: stab   = st.selectbox('Stability', ['A','B','C','D'],
                                              index=['A','B','C','D'].index(ph['stability']),
                                              format_func=lambda k: f'[{k}] {STABILITY_MODES[k][0]}',
                                              key=f'mp_s_{i}')
            alpha_i = st.number_input('Alpha (°)', -5.0, 15.0, ph['alpha'], 0.5, key=f'mp_al_{i}')

            rho_i, mu_i  = isa_atmosphere(altit)
            re_snap_i    = snap_re(rho_i * airspd * chord_mp / mu_i)
            cl_i         = compute_cl_phys(wt_mp, area_mp, airspd, altit, wu_mp)

            st.markdown(f"""
            <div style='margin:6px 0 6px 0;'>
                <span class='computed-badge'>Re = {re_snap_i:,.0f}</span>
                <span class='computed-badge'>CL target = {cl_i:.3f}</span>
                <span class='computed-badge'>ρ = {rho_i:.4f} kg/m³</span>
            </div></div>
            """, unsafe_allow_html=True)

            phases_out.append({'name': name.lower().replace(' ','_'),
                                're': re_snap_i, 'alpha': alpha_i,
                                'cl_target': round(cl_i, 4),
                                'stability': stab, 'cd_weight': cd_w,
                                'phase_weight': ph_w})
        else:
            with r[1]:
                re_raw_d = st.number_input('Re', 50000, 1000000, ph['re'], 50000,
                                            key=f'mp_re_{i}',
                                            help='Type or use arrows. Snaps to nearest trained Re.')
                re_d = snap_re(re_raw_d)
            with r[2]: alpha_d = st.number_input('Alpha (°)', -5.0, 15.0, ph['alpha'], 0.5, key=f'mp_ald_{i}')
            with r[3]: stab    = st.selectbox('Stability', ['A','B','C','D'],
                                               index=['A','B','C','D'].index(ph['stability']),
                                               format_func=lambda k: f'[{k}] {STABILITY_MODES[k][0]}',
                                               key=f'mp_sd_{i}')
            cl_col1, cl_col2 = st.columns([2, 1])
            with cl_col1:
                cl_d_slider = st.slider('CL target', -0.5, 2.5, ph['cl_target'], 0.05, key=f'mp_clt_{i}')
            with cl_col2:
                cl_d = st.number_input('CL precise', -0.5, 2.5, float(cl_d_slider), 0.01,
                                        key=f'mp_cltn_{i}', label_visibility='visible')
            st.markdown('</div>', unsafe_allow_html=True)

            phases_out.append({'name': name.lower().replace(' ','_'),
                                're': re_d, 'alpha': alpha_d, 'cl_target': cl_d,
                                'stability': stab, 'cd_weight': cd_w,
                                'phase_weight': ph_w})

    b1, b2, _ = st.columns([1, 1, 4])
    with b1:
        if st.button('＋ Add Phase') and len(st.session_state.phases_mp) < 5:
            st.session_state.phases_mp.append(
                {'name': f'Phase {len(st.session_state.phases_mp)+1}',
                 'airspeed':80.0,'altitude':1000.0,'re':500000,
                 'alpha':5.0,'cl_target':0.8,'stability':'B',
                 'cd_weight':0.5,'phase_weight':0.25})
            st.rerun()
    with b2:
        if st.button('－ Remove Last') and len(st.session_state.phases_mp) > 2:
            st.session_state.phases_mp.pop()
            st.rerun()

    # Geometric consistency
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("## Geometric Consistency Check")

    if len(phases_out) >= 2:
        p0, p1 = phases_out[0], phases_out[1]
        if p1['alpha'] != p0['alpha']:
            cla  = (p1['cl_target'] - p0['cl_target']) / (p1['alpha'] - p0['alpha'])
            cl0  = p0['cl_target'] - cla * p0['alpha']
            rows = [f"Linear polar ({p0['name']} + {p1['name']}):  "
                    f"CLα = {cla:.4f}/deg   CL₀ = {cl0:.4f}"]
            bad  = False
            for ph in phases_out[2:]:
                pred = cl0 + cla * ph['alpha']
                miss = pred - ph['cl_target']
                if abs(miss) > 0.05:
                    bad = True
                    rows.append(f"  ⚠ {ph['name']}: natural CL={pred:.3f}  "
                                f"target={ph['cl_target']:.3f}  miss={miss:+.3f}"
                                f"  → suggest ≈{pred:.2f}")
                else:
                    rows.append(f"  ✓ {ph['name']}: consistent  (miss={miss:+.3f})")

            note = ('Inconsistent targets — optimizer finds best compromise. '
                    'Conflicted phase will show ⚠ in results.'
                    if bad else '✓ All targets geometrically consistent.')
            st.markdown(f"""
            <div class='{"warn-box" if bad else "ok-box"}'>
                {'<br>'.join(rows)}<br><br>{note}
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    rc1, rc2 = st.columns([1, 2])
    with rc1: n_mp = st.slider('Iterations', 50, 300, 150, 10, key='mp_iter')
    with rc2:
        st.markdown(f"""
        <div class='metric-card' style='margin-top:24px;'>
            <div class='metric-label'>Phases</div>
            <div class='metric-value'>{len(phases_out)}</div>
            <div class='metric-sub'>
                {len(phases_out)} MLP calls/iter &nbsp;|&nbsp;
                {len(phases_out)*n_mp:,} total MLP predictions
            </div>
        </div>
        """, unsafe_allow_html=True)

    run_mp = st.button('▶  RUN MULTI-PHASE OPTIMIZER', use_container_width=True)

    if run_mp:
        raw_w   = [p['phase_weight'] for p in phases_out]
        total_w = sum(raw_w)
        profile = {
            'phases': [{k: v for k, v in p.items() if k != 'phase_weight'}
                       for p in phases_out],
            'phase_weights': [round(w/total_w, 4) for w in raw_w],
        }
        with open(FLIGHT_PROFILE, 'w') as f:
            json.dump(profile, f, indent=2)

        cmd = [str(OPTIMIZER_SCRIPT), '--mode', 'multi_phase',
               '--phases_file', str(FLIGHT_PROFILE), '--n_calls', str(n_mp)]
        st.markdown('<hr>', unsafe_allow_html=True)
        res = run_with_progress(cmd, 'Best composite')
        if res['code'] == 0:
            st.success('✓ Complete — navigate to 📊 Results')
        else:
            st.error('Optimizer error')
            st.code(res['log'][-3000:])


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == '📊  Results':

    st.markdown("# Results")

    # ── Run selector ─────────────────────────────────────────────────────────
    all_runs = get_all_runs()

    if not all_runs:
        st.markdown("""
        <div class='empty-state'>
            <div class='empty-state-icon'>📭</div>
            <div class='empty-state-text'>
                No runs found yet.<br><br>
                Run ⚙ Single Phase or ✈ Multi Phase first,<br>
                then come back here.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Build human-readable run labels
    def run_label(d: Path) -> str:
        name  = d.name  # e.g. run_20250307_143022_single_Re500000_a8.0
        parts = name.split('_', 2)   # ['run', '20250307', '143022_single_...']
        if len(parts) >= 3:
            date  = parts[1]           # 20250307
            rest  = parts[2]           # 143022_single_Re500000_a8.0
            time_ = rest.split('_')[0] # 143022
            desc  = '_'.join(rest.split('_')[1:])  # single_Re500000_a8.0
            return f'{date[:4]}-{date[4:6]}-{date[6:]}  {time_[:2]}:{time_[2:4]}  —  {desc}'
        return name

    run_options = {run_label(d): d for d in all_runs}
    selected_label = st.selectbox(
        '📁  Select run to view',
        list(run_options.keys()),
        index=0,
        help='Runs sorted newest first. Latest run is selected by default.',
    )
    OUT_DIR = run_options[selected_label]

    # Show run folder path as a small badge
    st.markdown(f"""
    <div style='font-family:Space Mono,monospace; font-size:0.62rem;
                color:#3a4a5a; margin-bottom:16px;'>
        📂 &nbsp; optimizer_results/{OUT_DIR.name}
    </div>
    """, unsafe_allow_html=True)

    report = (OUT_DIR / 'optimization_report.txt').read_text() \
             if (OUT_DIR / 'optimization_report.txt').exists() else ''
    is_multi = 'MULTI-PHASE' in report

    if not (OUT_DIR / 'best_airfoil.dat').exists():
        st.markdown("""
        <div class='warn-box'>
            This run folder exists but has no results yet —
            the optimizer may still be running, or it exited with an error.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Banner
    if is_multi:
        comp_v, feas_v = '—', '—'
        for line in report.splitlines():
            if 'Composite score' in line:
                try: comp_v = line.split(':')[1].strip().split()[0]
                except: pass
            if 'Feasible results' in line:
                try: feas_v = line.split(':')[1].strip()
                except: pass
        st.markdown(f"""
        <div style='display:flex; gap:14px; margin-bottom:24px; flex-wrap:wrap;'>
            <div class='metric-card' style='flex:1;'>
                <div class='metric-label'>Mode</div>
                <div class='metric-value' style='font-size:0.9rem;color:#81c784;'>
                    Multi-Phase</div>
            </div>
            <div class='metric-card' style='flex:1;'>
                <div class='metric-label'>Composite Score</div>
                <div class='metric-value'>{comp_v}</div>
                <div class='metric-sub'>lower is better</div>
            </div>
            <div class='metric-card' style='flex:1;'>
                <div class='metric-label'>Feasible Results</div>
                <div class='metric-value'>{feas_v}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        best = {'CL':'—','CD':'—','CM':'—','CL/CD':'—'}
        # Report format uses "  CL       : value" under "BEST AIRFOIL FOUND" section
        in_best_section = False
        for line in report.splitlines():
            if 'BEST AIRFOIL FOUND' in line:
                in_best_section = True
                continue
            if in_best_section:
                if line.strip().startswith('CL ') and ':' in line:
                    try: best['CL'] = line.split(':')[1].strip().split()[0]
                    except: pass
                elif line.strip().startswith('CD ') and ':' in line:
                    try: best['CD'] = line.split(':')[1].strip().split()[0]
                    except: pass
                elif line.strip().startswith('CM ') and ':' in line and 'feasible' not in line:
                    try: best['CM'] = line.split(':')[1].strip().split()[0]
                    except: pass
                elif line.strip().startswith('CL/CD') and ':' in line:
                    try: best['CL/CD'] = line.split(':')[1].strip().split()[0]
                    except: pass
                elif line.strip() == '' and best['CL'] != '—':
                    break  # end of best airfoil section
        metric_cards = ''.join(
            f"<div class='metric-card' style='flex:1;'>"
            f"<div class='metric-label'>Best {k}</div>"
            f"<div class='metric-value'>{v}</div></div>"
            for k, v in best.items()
        )
        st.markdown(
            f"<div style='display:flex; gap:14px; margin-bottom:24px; flex-wrap:wrap;'>"
            f"{metric_cards}</div>",
            unsafe_allow_html=True,
        )

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        '✈  Airfoil Shape',
        '🗺  SHAP Geometry Map',
        '📊  SHAP Bar Chart',
        '📋  Report',
    ])

    with tab1:
        # For multi-phase runs, show the combined all-phases figure if it exists
        all_phases_img = OUT_DIR / 'best_airfoil_all_phases.png'
        single_img     = OUT_DIR / 'best_airfoil.png'

        if is_multi and all_phases_img.exists():
            st.markdown("""
            <div style='font-family:Space Mono,monospace; font-size:0.68rem;
                        color:#5a6478; margin-bottom:10px;'>
                One optimized airfoil geometry evaluated across all flight phases.
                Each column shows the same airfoil operating at a different condition.
            </div>
            """, unsafe_allow_html=True)
            st.image(str(all_phases_img), use_container_width=True)
        elif single_img.exists():
            st.image(str(single_img), use_container_width=True)
        else:
            st.info('Airfoil shape plot not found.')

        dat = OUT_DIR / 'best_airfoil.dat'
        if dat.exists():
            st.markdown('<br>', unsafe_allow_html=True)
            st.download_button('⬇  Download best_airfoil.dat',
                               dat.read_bytes(), 'best_airfoil.dat', 'text/plain',
                               use_container_width=True)
            st.markdown("""
            <div class='info-box'>
                Load into XFOIL to validate predicted CL, CD, CM
                at each flight condition before use in design.
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div style='font-family:Space Mono,monospace; font-size:0.7rem;
                    color:#5a6478; margin-bottom:14px; line-height:1.9;'>
            Each airfoil is coloured by how much each geometric region influences
            the aerodynamic output. The airfoil is split into two surfaces:<br>
            &nbsp; • <b style='color:#c8cfe0'>Suction surface (top)</b> — the upper surface where air accelerates,
            pressure drops, and most lift is generated.<br>
            &nbsp; • <b style='color:#c8cfe0'>Pressure surface (bottom)</b> — the lower surface where pressure
            is higher, supporting the aircraft weight.<br><br>
            <b style='color:#1976d2'>Blue regions</b> — this part of the airfoil
            <b>increases</b> the aerodynamic output. &nbsp;
            <b style='color:#b71c1c'>Red regions</b> — this part
            <b>decreases</b> it.<br>
            The top 4 most influential chord regions are annotated with their
            contribution value and a plain-English region name.
        </div>
        """, unsafe_allow_html=True)

        # Parse CST from report — robust parser for both report formats
        cst_params, shap_data = None, load_shap_values(OUT_DIR)
        try:
            import ast as _ast
            lines = report.splitlines()
            upper, lower = None, None
            for i, line in enumerate(lines):
                stripped = line.strip()
                # Match either single-phase or multi-phase header
                if ('Upper surface' in stripped and
                        i + 1 < len(lines)):
                    candidate = lines[i + 1].strip()
                    if candidate.startswith('['):
                        try:
                            upper = _ast.literal_eval(candidate)
                        except Exception:
                            pass
                if ('Lower surface' in stripped and
                        'feasible' not in stripped and
                        i + 1 < len(lines)):
                    candidate = lines[i + 1].strip()
                    if candidate.startswith('['):
                        try:
                            lower = _ast.literal_eval(candidate)
                        except Exception:
                            pass
            if upper is not None and lower is not None:
                cst_params = list(upper) + list(lower)
        except Exception:
            pass

        # Read airfoil coordinates directly from dat file — always matches the displayed shape
        dat_path = OUT_DIR / 'best_airfoil.dat'
        x_dat, yu_dat, yl_dat = read_dat_xy(dat_path)
        xy_available = x_dat is not None

        if not xy_available and cst_params is not None:
            # Fallback to CST reconstruction only if dat read fails
            x_dat, yu_dat, yl_dat = cst_to_xy(cst_params)
            xy_available = True

        if not xy_available:
            st.info('Airfoil geometry not available. Re-run the optimizer.')
        elif shap_data is None:
            st.markdown("""
            <div class='warn-box'>
                shap_values.json not found.<br>
                SHAP explanation requires running the optimizer with SHAP enabled.<br>
                Showing plain airfoil geometry only.
            </div>
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(12, 4))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#ffffff')
            ax.fill_between(x_dat, yu_dat, yl_dat, color='#bbdefb', alpha=0.5)
            ax.plot(x_dat, yu_dat, color='#1565c0', lw=2)
            ax.plot(x_dat, yl_dat, color='#1565c0', lw=2)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title('Best Airfoil Geometry', color='#333333',
                         fontfamily='monospace', fontsize=10)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                        facecolor='#ffffff')
            plt.close(); buf.seek(0)
            st.image(buf, use_container_width=True)
        else:
            buf = make_shap_airfoil_figure(x_dat, yu_dat, yl_dat, shap_data)
            st.image(buf, use_container_width=True)
            st.markdown('<br>', unsafe_allow_html=True)
            st.download_button(
                '⬇  Download SHAP Geometry Map',
                buf.getvalue(), 'shap_geometry_map.png', 'image/png',
                use_container_width=True
            )

    with tab3:
        shap_data_t3 = load_shap_values(OUT_DIR)
        if shap_data_t3 is not None:
            # Regenerate bar chart with aerodynamic feature names
            feat_names = shap_data_t3.get('feature_names', [])
            shap_cl = np.array(shap_data_t3.get('shap_cl', []))
            shap_cd = np.array(shap_data_t3.get('shap_cd', []))
            shap_cm = np.array(shap_data_t3.get('shap_cm', []))

            # Map raw names to aerodynamic descriptions
            # Guard: ensure display_names length always matches feat_names
            display_names = [AERO_FEATURE_NAMES.get(n, n) for n in feat_names]

            fig, axes = plt.subplots(1, 3, figsize=(26, max(8, len(feat_names) * 0.48)))
            fig.patch.set_facecolor('#ffffff')
            titles  = ['CL — Lift Coefficient', 'CD — Drag Coefficient', 'CM — Pitching Moment']
            accents = ['#1565c0', '#e65100', '#6a1b9a']

            for ax, title, accent, shap_arr in zip(axes, titles, accents, [shap_cl, shap_cd, shap_cm]):
                ax.set_facecolor('#ffffff')
                if len(shap_arr) == 0:
                    ax.text(0.5, 0.5, 'No SHAP data', ha='center', va='center',
                            color='#888888', fontsize=9, transform=ax.transAxes)
                    ax.set_title(title, color=accent, fontfamily='monospace', fontsize=9)
                    continue

                # Trim both arrays to the same length to prevent FixedLocator mismatch
                n = min(len(shap_arr), len(display_names))
                shap_trimmed   = shap_arr[:n]
                labels_trimmed = display_names[:n]

                colors = ['#1565c0' if v >= 0 else '#c62828' for v in shap_trimmed]
                y_pos  = list(range(n))
                ax.barh(y_pos, shap_trimmed, color=colors, alpha=0.85, height=0.7)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels_trimmed, fontsize=8,
                                   color='#333333', fontfamily='monospace')
                ax.axvline(0, color='#aaaaaa', lw=1)
                ax.set_xlabel('SHAP value  (positive = increases output, negative = decreases)',
                              color='#444444', fontfamily='monospace', fontsize=7)
                ax.set_title(title, color=accent, fontfamily='monospace',
                             fontsize=10, fontweight='bold', pad=10)
                ax.tick_params(colors='#444444', labelsize=7)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#cccccc')
                ax.set_facecolor('#f8f9fa')

            fig.suptitle(
                'SHAP Feature Importance — which airfoil regions and conditions drive each output',
                color='#222222', fontfamily='monospace', fontsize=9, fontweight='bold'
            )
            plt.tight_layout()
            buf_t3 = io.BytesIO()
            plt.savefig(buf_t3, format='png', dpi=150, bbox_inches='tight',
                        facecolor='#ffffff')
            plt.close()
            buf_t3.seek(0)
            st.image(buf_t3, use_container_width=True)
            st.markdown('<br>', unsafe_allow_html=True)
            st.download_button(
                '⬇  Download SHAP Bar Chart',
                buf_t3.getvalue(), 'shap_bar_chart.png', 'image/png',
                use_container_width=True
            )
            st.markdown("""
            <div class='info-box'>
                Bars show SHAP contribution per airfoil region and flight condition.<br>
                <b style='color:#1976d2'>Blue (right)</b> = this feature increases the output. &nbsp;
                <b style='color:#b71c1c'>Red (left)</b> = this feature decreases it.<br>
                Feature names show which chord location (0–100%c) and surface is responsible.
                For multi-phase runs, cruise conditions are used as the reference point.
            </div>
            """, unsafe_allow_html=True)
        else:
            shap_img = OUT_DIR / 'shap_explanation.png'
            if shap_img.exists():
                st.image(str(shap_img), use_container_width=True)
            else:
                st.info('SHAP data not found. Install the shap package to enable.')

    with tab4:
        if report:
            st.code(report, language='text')
        rp = OUT_DIR / 'optimization_report.txt'
        if rp.exists():
            st.download_button('⬇  Download Report', rp.read_bytes(),
                               'optimization_report.txt', 'text/plain')
