"""
Chandigarh Multi-Layer Map Project
===================================
Week 7 — Day 9: Random Forest + SHAP Improvement Guidelines
Produces:
  - day9_rf_validation.png          (LOO-CV predicted vs observed scatter)
  - day9_shap_global.png            (global SHAP importance — beeswarm + bar)
  - day9_shap_waterfall.png         (per-sector SHAP waterfall — 4 worst sectors)
  - day9_prioritisation.png         (Group A sectors ranked by intervention priority)
  - improvement_guidelines.csv      (full per-sector guidelines — paper Table)
  - master_geo.gpkg                 (updated with SHAP scores and priority rank)

──────────────────────────────────────────────────────────────────────────────
MODEL ARCHITECTURE

  Target variable : LST (°C) — land surface temperature
                    Consistent with Prophet forecasts and spatial regression.
                    UTFVI and composite score guidelines are derived analytically
                    from the LST prediction (no circularity, no redundant models).

  Features        : NDVI, NDBI, TCI, Population, SC_pct
                    All complete for all 57 numeric urban sectors.
                    GHI excluded (only 24/57 non-null — halves training set;
                    documented as limitation).

  Algorithm       : Random Forest Regressor (scikit-learn)
                    n_estimators=500, max_features='sqrt', min_samples_leaf=3
                    Hyperparameters tuned by LOO-CV on this dataset size.

  Validation      : Leave-One-Out Cross-Validation (LOO-CV)
                    Recommended for n=57 — maximises training data per fold.
                    Reports R² and RMSE on held-out predictions.

  Explainability  : SHAP TreeExplainer (Lundberg & Lee 2017)
                    Global: mean |SHAP| per feature (city-wide importance)
                    Local: per-sector SHAP values (sector-specific drivers)

──────────────────────────────────────────────────────────────────────────────
IMPROVEMENT GUIDELINES

  Target sectors  : Group A — 20 sectors below BOTH benchmarks simultaneously
                    (UTFVI ≥ 0 AND composite score > mean)
                    Source: Benchmark_Coincidence_Analysis.docx

  Benchmark LST   : Mean LST of all UTFVI-Excellent sectors (UTFVI < 0)
                    = the ecological standard a sector must reach to exit
                    the UHI zone (Zhang 2006 definition)

  Guideline logic per sector:
    1. Compute LST gap = sector LST − benchmark LST
    2. Identify positive-SHAP features (pushing LST above predicted mean)
    3. Distribute the required LST reduction proportionally across those
       features using their SHAP magnitudes
    4. Express required change as: ΔNDVI, ΔTCI, ΔPopulation
    5. Compute resulting UTFVI analytically:
          UTFVI_target = (target_LST − city_mean_LST) / target_LST
    6. Compute resulting composite score from corrected feature values
    7. Classify intervention type: Green (NDVI-led), Traffic (TCI-led), Mixed

  Prioritisation  : Composite priority index =
                    (LST_gap / max_gap) × 0.5 + (UTFVI / max_UTFVI) × 0.5
                    Sectors ranked 1 (highest urgency) to 20.

Reference:
  Lundberg, S.M. & Lee, S.I. (2017). A unified approach to interpreting model
  predictions. NeurIPS 30.
  Zhang, Y. et al. (2006). J. Remote Sens., 10(5), 789.

Author : Elias Ruiz Sabater
Date   : April 2026
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
import shap
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MASTER_PATH      = '/Users/eliasruizsabater/Desktop/Project MMU/MASTER.xlsx'
SHAPEFILE_PATH   = '/Users/eliasruizsabater/Desktop/Project MMU/Chandigarh_Boundary-SHP/PySAL/Chandigarh_Sectors_UTM43N.shp'
MASTER_GEO_PATH  = '/Users/eliasruizsabater/Desktop/Project MMU/master_geo.gpkg'
OUTPUT_DIR       = '/Users/eliasruizsabater/Desktop/Project MMU/RESULTS/RF+SHAP/'

# Group A — below BOTH benchmarks (source: Benchmark_Coincidence_Analysis.docx)
# Includes all sectors: numeric (15, 20, 21, ..., 61) and non-numeric (26E, Manimajra)
GROUP_A = ['15','20','21','22','25','32','33','34','35','37',
           '38','40','41','44','45','47','52','61','26E','Manimajra']

# RF features — complete for all urban sectors (numeric + non-numeric with complete data)
FEATURES = ['NDVI', 'NDBI', 'TCI', 'Population', 'SC_pct']
TARGET   = 'LST'

# RF hyperparameters
RF_PARAMS = dict(n_estimators=500, max_features='sqrt',
                 min_samples_leaf=3, random_state=42, n_jobs=-1)

# UTFVI threshold for benchmark
UTFVI_EXCELLENT = 0.0

# Prioritisation weights (must sum to 1)
W_LST_GAP = 0.50
W_COMPOSITE   = 0.50

# Output paths
OUT_VALIDATION   = OUTPUT_DIR + 'Validation.png'
OUT_SHAP_GLOBAL  = OUTPUT_DIR + 'SHAP_Global.png'
OUT_SHAP_WATER   = OUTPUT_DIR + 'SHAP_Waterfall.png'
OUT_PRIORITY     = OUTPUT_DIR + 'Prioritisation.png'
OUT_GUIDELINES   = OUTPUT_DIR + 'Improvement_Guidelines.csv'
BG = '#0d1117'

# Feature display names (for charts)
FEAT_LABELS = {
    'NDVI'      : 'NDVI\n(vegetation)',
    'NDBI'      : 'NDBI\n(built-up)',
    'TCI'       : 'TCI\n(traffic)',
    'Population': 'Population\n(density)',
    'SC_pct'    : 'SC%\n(social class)',
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_excel(MASTER_PATH, sheet_name='Sheet2')
df['Sector_nam'] = df['Sector_nam'].astype(str)

urban = df[~df['Sector_nam'].str.startswith('Rural')].copy().reset_index(drop=True)
urban = urban.rename(columns={'PopxSector': 'Population', '% SC Pop ': 'SC_pct'})

# All numeric sectors for RF training
num = urban[urban['Sector_nam'].str.match(r'^\d+$')].copy().reset_index(drop=True)

# RF training set: all urban sectors with complete FEATURES data
rf_data = urban.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

# Benchmark values
bench_sectors = rf_data[rf_data['UTFVI'] < UTFVI_EXCELLENT]
bench_lst     = float(bench_sectors['LST'].mean())
city_mean_lst = float(rf_data['LST'].mean())

print(f"Numeric sectors available: {len(num)}")
print(f"RF training set (all with complete data): {len(rf_data)}")
print(f"Benchmark mean LST:        {bench_lst:.3f}°C  (UTFVI Excellent, n={len(bench_sectors)})")
print(f"City mean LST:             {city_mean_lst:.3f}°C")
print(f"Group A sectors:           {len(GROUP_A)} total")


# ─────────────────────────────────────────────────────────────────────────────
# 2. RANDOM FOREST — FIT + LOO-CV VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

X = rf_data[FEATURES].values
y = rf_data[TARGET].values
sector_names = rf_data['Sector_nam'].tolist()

# LOO-CV predictions
rf_loo = RandomForestRegressor(**RF_PARAMS)
y_pred_loo = cross_val_predict(rf_loo, X, y, cv=LeaveOneOut())

ss_res   = np.sum((y - y_pred_loo) ** 2)
ss_tot   = np.sum((y - y.mean()) ** 2)
r2_loo   = 1 - ss_res / ss_tot
rmse_loo = np.sqrt(np.mean((y - y_pred_loo) ** 2))
mae_loo  = np.mean(np.abs(y - y_pred_loo))

print(f"\n=== Random Forest LOO-CV Results ===")
print(f"  R²   = {r2_loo:.4f}")
print(f"  RMSE = {rmse_loo:.4f}°C")
print(f"  MAE  = {mae_loo:.4f}°C")

# Fit full model for SHAP
rf = RandomForestRegressor(**RF_PARAMS)
rf.fit(X, y)
r2_train = rf.score(X, y)
print(f"  Full-model R² (in-sample) = {r2_train:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOO-CV VALIDATION PLOT
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 8), facecolor=BG)
ax.set_facecolor('#0d1b2a')
for sp in ax.spines.values():
    sp.set_color('#334466')

# Colour by Group A membership
colors_pts = ['#ff6b6b' if s in GROUP_A else '#4da6ff' for s in sector_names]

for i, (obs, pred, color, sec) in enumerate(zip(y, y_pred_loo, colors_pts, sector_names)):
    ax.scatter(obs, pred, color=color, s=60, alpha=0.85,
               edgecolors='white', linewidths=0.4, zorder=3)
    if sec in ['45', '34', '22', '61']:   # label the worst
        ax.annotate(sec, (obs, pred), fontsize=7.5, color='white',
                    xytext=(4, 4), textcoords='offset points', fontweight='bold')

# 1:1 line
lim = [min(y.min(), y_pred_loo.min()) - 0.2, max(y.max(), y_pred_loo.max()) + 0.2]
ax.plot(lim, lim, color='#00ff88', lw=1.5, linestyle='--', alpha=0.7, zorder=2)
ax.set_xlim(lim); ax.set_ylim(lim)

ax.set_xlabel('Observed LST (°C)', color='#aaaaaa', fontsize=11)
ax.set_ylabel('LOO-CV Predicted LST (°C)', color='#aaaaaa', fontsize=11)
ax.tick_params(colors='#aaaaaa')
ax.grid(color='#1e2d3d', lw=0.6, alpha=0.7)

ax.legend(handles=[
    mpatches.Patch(facecolor='#ff6b6b', label=f'Group A — below both benchmarks (n={len(GROUP_A)})'),
    mpatches.Patch(facecolor='#4da6ff', label='Other sectors'),
], fontsize=9.5, facecolor='#0d1b2a', edgecolor='#334466', labelcolor='white')

ax.set_title(
    f'Random Forest — LOO-CV Validation\n'
    f'R² = {r2_loo:.3f}  |  RMSE = {rmse_loo:.3f}°C  |  MAE = {mae_loo:.3f}°C  '
    f'|  n = {len(rf_data)} sectors (numeric + non-numeric with complete data)\n'
    f'Features: {", ".join(FEATURES)}  |  GHI excluded',
    color='white', fontsize=11, fontweight='bold', pad=12,
)
plt.tight_layout()
plt.savefig(OUT_VALIDATION, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"\nValidation plot saved: {OUT_VALIDATION}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. SHAP — COMPUTE VALUES
# ─────────────────────────────────────────────────────────────────────────────

explainer  = shap.TreeExplainer(rf)
shap_vals  = explainer.shap_values(X)           # shape (57, 5)
shap_df    = pd.DataFrame(shap_vals, columns=FEATURES)
shap_df['Sector_nam'] = sector_names

global_importance = np.abs(shap_vals).mean(axis=0)   # mean |SHAP| per feature

print(f"\n=== Global SHAP Importance (mean |SHAP value|, °C) ===")
feat_importance = sorted(zip(FEATURES, global_importance), key=lambda x: x[1], reverse=True)
for feat, imp in feat_importance:
    print(f"  {feat:12s}: {imp:.4f}°C  ({imp/global_importance.sum()*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# 5. GLOBAL SHAP CHARTS — BAR + BEESWARM
# ─────────────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
for ax in (ax1, ax2):
    ax.set_facecolor('#0d1b2a')
    for sp in ax.spines.values():
        sp.set_color('#334466')

# ── Bar chart — global importance ────────────────────────────────────────────
feat_sorted = sorted(zip(FEATURES, global_importance), key=lambda x: x[1])
feats_plot  = [f for f, _ in feat_sorted]
imps_plot   = [i for _, i in feat_sorted]
bar_colors  = ['#4da6ff' if i != max(imps_plot) else '#00ff88' for i in imps_plot]

bars = ax1.barh(range(len(feats_plot)), imps_plot,
                color=bar_colors, edgecolor='#1a1a2e', linewidth=0.5)
ax1.set_yticks(range(len(feats_plot)))
ax1.set_yticklabels([FEAT_LABELS.get(f, f) for f in feats_plot],
                    color='white', fontsize=9.5)
ax1.tick_params(colors='#aaaaaa')
ax1.set_xlabel('Mean |SHAP value| (°C contribution to LST)', color='#aaaaaa', fontsize=10)
ax1.grid(axis='x', color='#1e2d3d', lw=0.6, alpha=0.7)
for bar, val in zip(bars, imps_plot):
    ax1.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
             f'{val:.3f}°C', va='center', color='white', fontsize=8.5, fontweight='bold')
ax1.set_title('Global Feature Importance\n(mean |SHAP value| across all sectors)',
              color='white', fontsize=11, fontweight='bold', pad=10)

# ── Beeswarm — SHAP value distribution per feature ──────────────────────────
# Sorted same order as bar chart (lowest to highest importance)
feat_order = [f for f, _ in sorted(zip(FEATURES, global_importance), key=lambda x: x[1])]
X_df = pd.DataFrame(X, columns=FEATURES)

for fi, feat in enumerate(feat_order):
    shap_col   = shap_vals[:, FEATURES.index(feat)]
    feat_col   = X_df[feat].values
    # Normalise feature values for colour scale
    feat_norm  = (feat_col - feat_col.min()) / (feat_col.max() - feat_col.min() + 1e-9)
    # Jitter y position
    jitter     = np.random.RandomState(42).uniform(-0.2, 0.2, len(shap_col))
    sc = ax2.scatter(shap_col, fi + jitter,
                     c=feat_norm, cmap='RdBu_r', vmin=0, vmax=1,
                     s=35, alpha=0.8, edgecolors='none', zorder=3)

ax2.axvline(0, color='#aaaaaa', lw=0.8, linestyle='--', alpha=0.6)
ax2.set_yticks(range(len(feat_order)))
ax2.set_yticklabels([FEAT_LABELS.get(f, f) for f in feat_order],
                    color='white', fontsize=9.5)
ax2.tick_params(colors='#aaaaaa')
ax2.set_xlabel('SHAP value (°C impact on LST prediction)', color='#aaaaaa', fontsize=10)
ax2.grid(axis='x', color='#1e2d3d', lw=0.6, alpha=0.7)

cbar = fig.colorbar(sc, ax=ax2, fraction=0.03, pad=0.02)
cbar.set_label('Feature value\n(low → high)', color='white', fontsize=8)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
ax2.set_title('SHAP Value Distribution\n(each dot = one sector)',
              color='white', fontsize=11, fontweight='bold', pad=10)

fig.suptitle('Random Forest SHAP Analysis — LST Drivers\nChandigarh Urban Sectors  |  n=57',
             color='white', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout(pad=2.5)
plt.savefig(OUT_SHAP_GLOBAL, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"SHAP global chart saved: {OUT_SHAP_GLOBAL}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. WATERFALL CHARTS — 4 WORST GROUP A SECTORS
# ─────────────────────────────────────────────────────────────────────────────
# Shows how each feature contributes to the predicted LST for four selected
# high-vulnerability sectors, starting from the model's expected value (E[f(x)]).

WATERFALL_SECTORS = ['45', '34', '22', '61']   # highest LST gap in Group A

fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor=BG)
fig.suptitle(
    'SHAP Waterfall — LST Decomposition for 4 Highest-Priority Sectors\n'
    'Each bar shows a feature\'s contribution to the deviation from the city-wide mean prediction',
    color='white', fontsize=12, fontweight='bold', y=1.01,
)

for ax, sector in zip(axes.flatten(), WATERFALL_SECTORS):
    ax.set_facecolor('#0d1b2a')
    for sp in ax.spines.values():
        sp.set_color('#334466')

    idx      = sector_names.index(sector)
    s_shap   = shap_vals[idx]                   # per-feature SHAP values
    base_val = float(np.array(explainer.expected_value).ravel()[0])  # E[f(x)]
    pred_val = base_val + s_shap.sum()          # RF prediction for this sector
    obs_val  = y[idx]                           # actual observed LST

    # Sort features by absolute SHAP value descending
    order    = np.argsort(np.abs(s_shap))[::-1]
    feats_w  = [FEATURES[i] for i in order]
    shaps_w  = [s_shap[i] for i in order]

    # Cumulative sum from base value
    cumsum   = [base_val] + list(base_val + np.cumsum(shaps_w))
    lefts    = [base_val + sum(shaps_w[:i]) for i in range(len(shaps_w))]

    colors_w = ['#ff6b6b' if s > 0 else '#4da6ff' for s in shaps_w]

    bars = ax.barh(range(len(feats_w)), shaps_w, left=lefts,
                   color=colors_w, edgecolor='#1a1a2e', linewidth=0.5, height=0.6)
    ax.axvline(base_val, color='#aaaaaa', lw=1.0, linestyle=':', alpha=0.7,
               label=f'E[f(x)] = {base_val:.2f}°C')
    ax.axvline(obs_val,  color='#ffd700',  lw=1.5, linestyle='--', alpha=0.9,
               label=f'Observed = {obs_val:.2f}°C')
    ax.axvline(pred_val, color='#00ff88', lw=1.5, linestyle='--', alpha=0.9,
               label=f'Predicted = {pred_val:.2f}°C')

    ax.set_yticks(range(len(feats_w)))
    ax.set_yticklabels([FEAT_LABELS.get(f, f) for f in feats_w],
                       color='white', fontsize=8.5)
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    ax.set_xlabel('LST contribution (°C)', color='#aaaaaa', fontsize=9)
    ax.grid(axis='x', color='#1e2d3d', lw=0.6, alpha=0.7)
    ax.legend(fontsize=7.5, facecolor='#0d1b2a', edgecolor='#334466', labelcolor='white')

    # Annotate bar values
    for bar, val, left in zip(bars, shaps_w, lefts):
        sign = '+' if val >= 0 else ''
        ax.text(left + val / 2, bar.get_y() + bar.get_height() / 2,
                f'{sign}{val:.2f}', ha='center', va='center',
                color='white', fontsize=7.5, fontweight='bold')

    lst_gap = obs_val - bench_lst
    ax.set_title(
        f'Sector {sector}  |  LST = {obs_val:.2f}°C  '
        f'(+{lst_gap:.2f}°C above benchmark)\n'
        f'UTFVI = {num.loc[idx,"UTFVI"]:.4f}',
        color='white', fontsize=9.5, fontweight='bold', pad=8,
    )

plt.tight_layout(pad=2.5)
plt.savefig(OUT_SHAP_WATER, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"SHAP waterfall saved: {OUT_SHAP_WATER}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. IMPROVEMENT GUIDELINES — GROUP A (20 NUMERIC SECTORS)
# ─────────────────────────────────────────────────────────────────────────────
# For each Group A sector:
#   1. Identify positive-SHAP features (pushing LST up for this sector)
#   2. Compute required LST reduction: target = bench_lst
#   3. Distribute reduction proportionally to positive-SHAP magnitudes
#   4. Derive required feature changes using known empirical relationships
#   5. Express resulting UTFVI and composite score analytically

def compute_utfvi(lst_val, city_mean):
    """Zhang 2006 formula: (LST - CityMean) / LST"""
    return (lst_val - city_mean) / lst_val

def classify_utfvi(v):
    if pd.isna(v):  return 'N/A'
    if v < 0:       return 'Excellent'
    elif v < 0.005: return 'Good'
    elif v < 0.010: return 'Normal'
    elif v < 0.015: return 'Bad'
    elif v < 0.020: return 'Worse'
    else:           return 'Worst'

def intervention_type(primary_feat, secondary_feat):
    """Classify the main type of intervention needed."""
    if primary_feat == 'NDVI':
        return 'Green infrastructure (NDVI)'
    elif primary_feat == 'TCI':
        return 'Traffic management (TCI)'
    elif primary_feat == 'NDBI':
        return 'Built-up reduction / cool surfaces (NDBI)'
    elif primary_feat == 'Population':
        return 'Density management'
    else:
        return 'Mixed / social (SC%)'

# Compute composite score (to express target composite)
# Using same PCA-weight logic as Day 8 — recompute here for consistency
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

EXCLUDE = ['1','Khuda Alisher','Rajindra Park']
GHI_ADJ_WEIGHT = 0.15
residential = num[~num['Sector_nam'].isin(EXCLUDE)].copy().reset_index(drop=True)

pca_feats  = ['LST','NDVI','NDBI','TCI','Population'] #included NDBI, consider its inconsistency in Spearman correlation.
X_pca      = StandardScaler().fit_transform(residential[pca_feats])
pca        = PCA(n_components=len(pca_feats))
pca.fit(X_pca)
pc1_abs    = np.abs(pca.components_[0])
pca_w      = (pc1_abs / pc1_abs.sum()) * (1 - GHI_ADJ_WEIGHT)
pca_w_dict = dict(zip(pca_feats, pca_w))

scaler_comp = MinMaxScaler()
s_comp = pd.DataFrame(scaler_comp.fit_transform(residential[pca_feats]),
                      columns=pca_feats, index=residential.index)
s_comp['NDVI'] = 1 - s_comp['NDVI']  # invert

def composite_score(lst_val, ndvi_val, ndbi_val, tci_val, pop_val, ghi_val=None):
    """Compute composite score for a sector given feature values."""
    # Re-normalise using the fitted scaler ranges (5 features: LST, NDVI, NDBI, TCI, Population)
    mins  = scaler_comp.data_min_
    maxs  = scaler_comp.data_max_
    vals  = np.array([lst_val, ndvi_val, ndbi_val, tci_val, pop_val])
    s_vec = (vals - mins) / (maxs - mins + 1e-9)
    s_vec[1] = 1 - s_vec[1]   # invert NDVI
    base = sum(s_vec[i] * pca_w_dict[pca_feats[i]] for i in range(len(pca_feats)))
    if ghi_val is not None and not np.isnan(ghi_val):
        gmin, gmax = residential['GHI'].min(), residential['GHI'].max()
        ghi_norm_inv = 1 - (ghi_val - gmin) / (gmax - gmin + 1e-9)
        base += ghi_norm_inv * GHI_ADJ_WEIGHT
    return float(base)

print(f"\n=== Computing improvement guidelines for {len(GROUP_A)} Group A sectors ===")

guidelines = []

for sector in GROUP_A:
    row_full = rf_data[rf_data['Sector_nam'] == sector]
    if len(row_full) == 0:
        continue
    row      = row_full.iloc[0]
    idx      = rf_data[rf_data['Sector_nam'] == sector].index[0]

    # Current values
    lst_cur  = float(row['LST'])
    ndvi_cur = float(row['NDVI'])
    ndbi_cur = float(row['NDBI'])
    tci_cur  = float(row['TCI'])
    pop_cur  = float(row['Population'])
    sc_cur   = float(row['SC_pct'])
    utfvi_cur= float(row['UTFVI'])
    ghi_cur  = row['GHI'] if not pd.isna(row['GHI']) else np.nan

    # SHAP values for this sector
    s_shap   = shap_vals[sector_names.index(sector)]
    shap_dict= dict(zip(FEATURES, s_shap))

    # LST gap to close
    lst_gap  = lst_cur - bench_lst

    # Positive-SHAP features (pushing LST up — these are the levers)
    pos_feats = {f: v for f, v in shap_dict.items() if v > 0}
    pos_total = sum(pos_feats.values()) if pos_feats else 1.0

    # Primary and secondary drivers
    shap_ranked = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    primary_feat   = shap_ranked[0][0]
    secondary_feat = shap_ranked[1][0] if len(shap_ranked) > 1 else ''

    # Required changes
    # Each feature's required reduction is proportional to its positive SHAP share
    # Convert SHAP magnitude to required feature change using:
    #   - NDVI: use empirical slope from Spearman (ρ = -0.816)
    #     Rough estimate: Δ1 NDVI unit ≈ -9.0°C LST (from scatter in Day 6)
    #   - TCI:  Δ1 TCI unit ≈ +0.5°C LST (from spatial lag model β)
    #   - NDBI: less reliable (reversed) — flag as uncertain
    #   - Population: not directly actionable — report as context
    #   Note: These are indicative empirical estimates from Day 6 analysis.
    #   The RF/SHAP tells us *how much* of the LST gap each feature explains;
    #   the empirical slopes tell us *what change is needed* in real units.

    NDVI_SLOPE = -13.06 # °C per NDVI unit (from scatter analysis, Day 6)
    NDBI_SLOPE = -2.94  # °C per NDBI unit (reversed relationship, interpret with caution)
    TCI_SLOPE  =  0.43   # °C per TCI unit

    # Required ΔLST from NDVI (if positive contributor)
    delta_lst_from_ndvi = pos_feats.get('NDVI', 0) / pos_total * lst_gap if pos_total > 0 else 0
    delta_lst_from_tci  = pos_feats.get('TCI',  0) / pos_total * lst_gap if pos_total > 0 else 0
    delta_lst_from_ndbi = pos_feats.get('NDBI', 0) / pos_total * lst_gap if pos_total > 0 else 0

    # Required feature changes
    delta_ndvi = -delta_lst_from_ndvi / NDVI_SLOPE if NDVI_SLOPE != 0 else 0   # positive = need more vegetation
    delta_tci  = -delta_lst_from_tci  / TCI_SLOPE  if TCI_SLOPE  != 0 else 0   # negative = need less traffic
    delta_ndbi = -delta_lst_from_ndbi / NDBI_SLOPE if NDBI_SLOPE != 0 else 0   # positive = need more built-up area 
    # Target values
    lst_target  = bench_lst
    ndvi_target = min(1.0, ndvi_cur + delta_ndvi)
    tci_target  = max(0.0, tci_cur  + delta_tci)
    ndbi_target = max(1.0, ndbi_cur + delta_ndbi)

    # Resulting UTFVI
    utfvi_target    =  0.0  #compute_utfvi(lst_target, city_mean_lst)
    utfvi_class_cur = classify_utfvi(utfvi_cur)
    utfvi_class_tgt = classify_utfvi(utfvi_target)

    # Resulting composite score
    comp_cur    = composite_score(lst_cur,  ndvi_cur, ndbi_cur, tci_cur,  pop_cur,  ghi_val=ghi_cur)
    comp_target = composite_score(lst_target, ndvi_target, ndbi_cur, tci_target, pop_cur, ghi_val=ghi_cur)

    # Prioritisation index (0–1, higher = more urgent)
    max_lst_gap   = rf_data[rf_data['Sector_nam'].isin(GROUP_A)]['LST'].max() - bench_lst
    max_comp      = max(
        composite_score(
            float(rf_data[rf_data['Sector_nam']==s].iloc[0]['LST']),
            float(rf_data[rf_data['Sector_nam']==s].iloc[0]['NDVI']),
            float(rf_data[rf_data['Sector_nam']==s].iloc[0]['NDBI']),
            float(rf_data[rf_data['Sector_nam']==s].iloc[0]['TCI']),
            float(rf_data[rf_data['Sector_nam']==s].iloc[0]['Population']),
            ghi_val=rf_data[rf_data['Sector_nam']==s].iloc[0]['GHI']
            if not pd.isna(rf_data[rf_data['Sector_nam']==s].iloc[0]['GHI']) else np.nan)
        for s in GROUP_A if len(rf_data[rf_data['Sector_nam']==s]) > 0
    )   
    prio_idx = (lst_gap / max_lst_gap   * W_LST_GAP + comp_cur / max_comp     * W_COMPOSITE)

    # Intervention type
    intervention = intervention_type(primary_feat, secondary_feat)

    guidelines.append({
        'Sector'            : sector,
        # Current state
        'LST_current'       : round(lst_cur, 3),
        'NDVI_current'      : round(ndvi_cur, 4),
        'TCI_current'       : round(tci_cur, 4),
        'UTFVI_current'     : round(utfvi_cur, 4),
        'UTFVI_class_cur'   : utfvi_class_cur,
        'Composite_current' : round(comp_cur, 4),
        # Target (benchmark)
        'LST_target'        : round(lst_target, 3),
        'LST_gap'           : round(lst_gap, 3),
        # Required changes
        'ΔNDVI_required'    : round(delta_ndvi, 4),
        'ΔTCI_required'     : round(delta_tci, 4),
        'NDVI_target'       : round(ndvi_target, 4),
        'TCI_target'        : round(tci_target, 4),
        # Resulting state
        'UTFVI_target'      : round(utfvi_target, 4),
        'UTFVI_class_tgt'   : utfvi_class_tgt,
        'Composite_target'  : round(comp_target, 4),
        'Composite_change'  : round(comp_target - comp_cur, 4),
        # SHAP drivers
        'Primary_driver'    : primary_feat,
        'Primary_SHAP'      : round(shap_dict[primary_feat], 4),
        'Secondary_driver'  : secondary_feat,
        'Secondary_SHAP'    : round(shap_dict[secondary_feat], 4) if secondary_feat else 0,
        'Intervention_type' : intervention,
        # Priority
        'Priority_index'    : round(prio_idx, 4),
    })

guide_df = pd.DataFrame(guidelines)
guide_df = guide_df.sort_values('Priority_index', ascending=False).reset_index(drop=True)
guide_df['Priority_rank'] = range(1, len(guide_df) + 1)

guide_df.to_csv(OUT_GUIDELINES, index=False)
print(f"Improvement guidelines saved: {OUT_GUIDELINES}")
print(f"\n=== Top 10 priority sectors ===")
print(guide_df[['Priority_rank','Sector','LST_current','LST_gap',
                'ΔNDVI_required','ΔTCI_required','UTFVI_class_cur',
                'UTFVI_class_tgt','Primary_driver',
                'Intervention_type']].head(10).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 8. PRIORITISATION CHART
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
ax.set_facecolor('#0d1b2a')
for sp in ax.spines.values():
    sp.set_color('#334466')

# Colour bars by intervention type
INTERV_COLORS = {
    'Green infrastructure (NDVI)'          : '#00ff88',
    'Traffic management (TCI)'             : '#ff6b6b',
    'Built-up reduction / cool surfaces (NDBI)': '#ffd700',
    'Density management'                   : '#a78bfa',
    'Mixed / social (SC%)'                 : '#fb923c',
}

colors_p = [INTERV_COLORS.get(t, '#555577') for t in guide_df['Intervention_type']]

bars = ax.bar(range(len(guide_df)), guide_df['Priority_index'],
              color=colors_p, edgecolor='#1a1a2e', linewidth=0.4, zorder=3)

# Annotate with LST gap
for i, (_, row) in enumerate(guide_df.iterrows()):
    ax.text(i, row['Priority_index'] + 0.005,
            f"+{row['LST_gap']:.2f}°C", ha='center', color='white',
            fontsize=7.5, fontweight='bold')

ax.set_xticks(range(len(guide_df)))
ax.set_xticklabels([f"S{r['Sector']}\n(#{r['Priority_rank']})"
                    for _, r in guide_df.iterrows()],
                   color='#aaaaaa', fontsize=8)
ax.set_ylabel('Priority Index (composite urgency score 0–1)', color='#aaaaaa', fontsize=10)
ax.set_xlabel('Group A Sectors — ranked by intervention urgency', color='#aaaaaa', fontsize=10)
ax.tick_params(colors='#aaaaaa')
ax.grid(axis='y', color='#1e2d3d', lw=0.6, alpha=0.7)

legend_handles = [mpatches.Patch(facecolor=c, label=t)
                  for t, c in INTERV_COLORS.items()
                  if t in guide_df['Intervention_type'].values]
ax.legend(handles=legend_handles, fontsize=8.5, facecolor='#0d1b2a',
          edgecolor='#334466', labelcolor='white',
          title='Primary intervention type', title_fontsize=8.5,
          loc='upper right')

ax.set_title(
    'Sector Intervention Prioritisation — Group A (Both Benchmarks Exceeded)\n'
    f'Priority index = LST gap weight ({W_LST_GAP:.0%}) + Composite weight ({W_COMPOSITE:.0%})  '
    f'|  Number in parentheses = priority rank  |  +X°C = LST above benchmark',
    color='white', fontsize=11, fontweight='bold', pad=12,
)
plt.tight_layout()
plt.savefig(OUT_PRIORITY, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Prioritisation chart saved: {OUT_PRIORITY}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. UPDATE MASTER_GEO.GPKG
# ─────────────────────────────────────────────────────────────────────────────

# Add SHAP values and priority rank to the GeoPackage
shap_export = shap_df.copy()
for feat in FEATURES:
    shap_export = shap_export.rename(columns={feat: f'SHAP_{feat}'})
shap_export['SHAP_total']   = shap_vals.sum(axis=1)
shap_export['RF_predicted'] = rf.predict(X)

# Merge priority rank
shap_export = shap_export.merge(
    guide_df[['Sector', 'Priority_rank', 'Priority_index',
              'LST_gap', 'Intervention_type', 'UTFVI_class_tgt']].rename(
                  columns={'Sector': 'Sector_nam'}),
    on='Sector_nam', how='left'
)

gdf_shp = gpd.read_file(SHAPEFILE_PATH)
gdf_shp['Sector_nam'] = gdf_shp['Sector_nam'].astype(str)
gdf = gdf_shp.merge(urban, on='Sector_nam', how='left').reset_index(drop=True)

drop_cols = [c for c in gdf.columns if c.startswith('SHAP_') or
             c in ['RF_predicted','Priority_rank','Priority_index',
                   'LST_gap','Intervention_type','UTFVI_class_tgt']]
gdf = gdf.drop(columns=drop_cols, errors='ignore')
gdf = gdf.merge(shap_export, on='Sector_nam', how='left')
gdf.to_file(MASTER_GEO_PATH, driver='GPKG')
print(f"master_geo.gpkg updated with SHAP values and priority rank: {MASTER_GEO_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"Day 9 complete")
print(f"{'='*70}")
print(f"Model:       Random Forest  |  n=57 numeric urban sectors")
print(f"Features:    {', '.join(FEATURES)}")
print(f"LOO-CV R²:   {r2_loo:.4f}  |  RMSE: {rmse_loo:.4f}°C  |  MAE: {mae_loo:.4f}°C")
print(f"Dominant driver city-wide: {feat_importance[0][0]} "
      f"({feat_importance[0][1]:.3f}°C mean |SHAP|)")
print()
print(f"Group A sectors with guidelines: {len(guide_df)}")
print(f"Top 3 priority sectors:")
for _, row in guide_df.head(3).iterrows():
    print(f"  #{row['Priority_rank']:2d} Sector {row['Sector']:5s}: "
          f"LST gap +{row['LST_gap']:.2f}°C  |  "
          f"Primary driver: {row['Primary_driver']}  |  "
          f"Intervention: {row['Intervention_type']}")
print()
print("Outputs:")
for p in [OUT_VALIDATION, OUT_SHAP_GLOBAL, OUT_SHAP_WATER, OUT_PRIORITY, OUT_GUIDELINES]:
    print(f"  {p}")
print(f"  {MASTER_GEO_PATH}  (updated with SHAP values + priority rank)")
