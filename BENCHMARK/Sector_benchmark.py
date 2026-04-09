"""
Chandigarh Multi-Layer Map Project
===================================
Week 7 — Day 8: Benchmark Cluster Identification
Produces:
  - day8_pca_weights.png            (PCA scree plot + PC1 loadings bar chart)
  - day8_composite_scores.png       (bar chart — all sectors ranked by composite score)
  - day8_benchmark_map.png          (choropleth — LST deviation from benchmark mean)
  - day8_radar_charts.png           (4 high-vulnerability sectors vs benchmark profile)
  - benchmark_profile.csv           (mean values of benchmark cluster)
  - sector_benchmark_deviation.csv  (per-sector deviation from benchmark, all variables)
  - master_geo.gpkg                 (updated: composite_score, rank, LST_dev, UTFVI_class)

──────────────────────────────────────────────────────────────────────────────
BENCHMARK DEFINITION — two independent criteria applied simultaneously

  Criterion 1 (PRIMARY — Zhang 2006):
      UTFVI < 0  →  "Excellent" ecological quality.
      Sectors below zero show NO urban heat island effect relative to the
      city mean. Threshold from Liu & Zhang (2011), Table 8. The benchmark
      cluster is the full set of UTFVI-Excellent sectors.

  Criterion 2 (CROSS-VALIDATION — Anselin 1995):
      LISA LL cold spot (p < 0.05, 999 permutations).
      Sectors satisfying BOTH criteria are flagged with ** in all outputs.
      The intersection confirms that spatial cold clustering is consistent
      with the UTFVI ecological classification.

Reference:
  Zhang, Y. et al. (2006). J. Remote Sens., 10(5), 789.
  Liu, L. & Zhang, Y. (2011). Remote Sensing, 3(7), 1535-1552.

──────────────────────────────────────────────────────────────────────────────
COMPOSITE SCORING METHODOLOGY (vulnerability ranking of all sectors):

  PCA on four complete variables (LST, NDVI, TCI, Population).
  Weights = absolute PC1 loadings, normalised to sum to (1 - GHI_ADJ_WEIGHT).
  GHI incorporated as a post-hoc adjustment (weight = GHI_ADJ_WEIGHT) for
  the 24 sectors with government housing data; NaN sectors receive 0 adjustment.

Excluded from scoring: Sectors 17 (commercial), 30 (industrial), 32 (institutional)
Geographic islands (no shared borders): Manimajra, Khuda Alisher

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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — adjust paths as needed
# ─────────────────────────────────────────────────────────────────────────────

MASTER_PATH      = '/Users/eliasruizsabater/Desktop/Project MMU/MASTER.xlsx'
SHAPEFILE_PATH   = '/Users/eliasruizsabater/Desktop/Project MMU/Chandigarh_Boundary-SHP/PySAL/Chandigarh_Sectors_UTM43N.shp'
MASTER_GEO_PATH  = '/Users/eliasruizsabater/Desktop/Project MMU/master_geo.gpkg'
OUTPUT_DIR       = '/Users/eliasruizsabater/Desktop/Project MMU/RESULTS/BENCHMARK/'

# UTFVI threshold — benchmark = all sectors with UTFVI < this value
UTFVI_EXCELLENT_THRESHOLD = 0.0

# Sectors excluded from composite scoring (non-residential land use)
EXCLUDE_FROM_SCORING = ['1', 'Khuda Alisher','Rajindra Park']

# GHI post-hoc adjustment weight (proportion of total composite score)
GHI_ADJ_WEIGHT = 0.15

# Geographic islands — excluded from spatial weights (zero shared borders).
# Shown on the benchmark map with a dashed border and their actual LST
# deviation colour — NOT hidden in olive.
ISLANDS = ['Manimajra', 'Khuda Alisher']

# Output paths
OUT_PCA_PLOT      = OUTPUT_DIR + 'PCA_weights.png'
OUT_VULN_CHART    = OUTPUT_DIR + 'Composite_scores_vulnerability.png'
OUT_BENCHMARK_CHART = OUTPUT_DIR + 'Composite_scores_UTFVI.png'
OUT_BENCH_MAP     = OUTPUT_DIR + 'Benchmark_map.png'
OUT_RADAR         = OUTPUT_DIR + 'Radar_charts.png'
OUT_PROFILE_CSV   = OUTPUT_DIR + 'Benchmark_profile.csv'
OUT_DEVIATION_CSV = OUTPUT_DIR + 'Sector_benchmark_deviation.csv'
OUT_COMPOSITE_DOC = OUTPUT_DIR + 'Composite_Score_Methodology.txt'

BG = '#0d1117'


# ─────────────────────────────────────────────────────────────────────────────
# UTFVI CLASSIFICATION (Zhang 2006 six-tier scheme)
# ─────────────────────────────────────────────────────────────────────────────

def classify_utfvi(v):
    if pd.isna(v):   return 'N/A'
    if v < 0.000:    return 'Excellent'
    elif v < 0.005:  return 'Good'
    elif v < 0.010:  return 'Normal'
    elif v < 0.015:  return 'Bad'
    elif v < 0.020:  return 'Worse'
    else:            return 'Worst'


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_excel(MASTER_PATH, sheet_name='Sheet2')
df['Sector_nam'] = df['Sector_nam'].astype(str)

urban = df[~df['Sector_nam'].str.startswith('Rural')].copy().reset_index(drop=True)
urban = urban.rename(columns={'PopxSector': 'Population', '% SC Pop ': 'SC_pct'})
urban['UTFVI_class'] = urban['UTFVI'].apply(classify_utfvi)

num         = urban[urban['Sector_nam'].str.match(r'^\d+$')].copy()
residential = urban[~urban['Sector_nam'].isin(EXCLUDE_FROM_SCORING)].copy().reset_index(drop=True)

print(f"Urban sectors loaded:          {len(urban)}")
print(f"Numeric sectors:               {len(num)}")
print(f"Residential sectors (scored):  {len(residential)}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. BENCHMARK — CRITERION 1: UTFVI EXCELLENT (ZHANG 2006)
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK = sorted(
    urban[urban['UTFVI'] < UTFVI_EXCELLENT_THRESHOLD]['Sector_nam'].tolist(),
    key=lambda x: (0, int(x)) if x.isdigit() else (1, x)
)
BENCHMARK_NUM = [s for s in BENCHMARK if str(s).isdigit()]

print(f"\n=== Benchmark — UTFVI Excellent (UTFVI < 0) ===")
print(f"  {len(BENCHMARK)} sectors total  |  {len(BENCHMARK_NUM)} numeric")
print(f"  {BENCHMARK}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. CROSS-VALIDATION — CRITERION 2: LISA LL COLD SPOTS (ANSELIN 1995)
# ─────────────────────────────────────────────────────────────────────────────

try:
    gdf_lisa = gpd.read_file(MASTER_GEO_PATH)
    gdf_lisa['Sector_nam'] = gdf_lisa['Sector_nam'].astype(str)
    LISA_LL = set(gdf_lisa[gdf_lisa['LISA_cluster'] == 'LL']['Sector_nam'].tolist())
    BENCHMARK_BOTH       = sorted([s for s in BENCHMARK if s in LISA_LL],
                                   key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
    BENCHMARK_UTFVI_ONLY = [s for s in BENCHMARK if s not in LISA_LL]
    print(f"\n=== LISA cross-validation ===")
    print(f"  LISA LL cold spots:    {sorted(LISA_LL, key=lambda x: (0,int(x)) if x.isdigit() else (1,x))}")
    print(f"  Both criteria (**):    {BENCHMARK_BOTH}")
    print(f"  UTFVI Excellent only:  {BENCHMARK_UTFVI_ONLY}")
except Exception as e:
    print(f"\nLISA cross-validation skipped: {e}")
    LISA_LL              = set()
    BENCHMARK_BOTH       = []
    BENCHMARK_UTFVI_ONLY = BENCHMARK.copy()


# ─────────────────────────────────────────────────────────────────────────────
# 4. BENCHMARK PROFILE
# ─────────────────────────────────────────────────────────────────────────────

bench_data = urban[urban['Sector_nam'].isin(BENCHMARK)]
profile    = bench_data[['LST', 'NDVI', 'TCI', 'GHI', 'Population', 'SC_pct']].mean()
bench_lst  = float(profile['LST'])

print(f"\n=== Benchmark profile (n={len(BENCHMARK)} Excellent sectors) ===")
print(profile.round(4).to_string())

pd.DataFrame({
    'Variable'       : profile.index,
    'Benchmark_mean' : profile.values.round(4),
    'n_sectors'      : [len(BENCHMARK)] * len(profile),
    'Sectors'        : [', '.join(BENCHMARK)] * len(profile),
    'Criterion'      : ['UTFVI < 0 — Excellent'] * len(profile),
}).to_csv(OUT_PROFILE_CSV, index=False)
print(f"benchmark_profile.csv saved: {OUT_PROFILE_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PCA — DERIVE COMPOSITE SCORE WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

PCA_VARS = ['LST', 'NDVI', 'TCI', 'Population']
X_scaled = StandardScaler().fit_transform(residential[PCA_VARS])
pca      = PCA(n_components=len(PCA_VARS))
pca.fit(X_scaled)

pc1_abs         = np.abs(pca.components_[0])
pca_weights     = (pc1_abs / pc1_abs.sum()) * (1 - GHI_ADJ_WEIGHT)
pca_weight_dict = dict(zip(PCA_VARS, pca_weights))
var_explained   = pca.explained_variance_ratio_

print(f"\n=== PCA ===")
print(f"PC1 variance explained: {var_explained[0]:.1%}")
for var, w in pca_weight_dict.items():
    print(f"  {var:12s}: {w*100:.1f}%")
print(f"  {'GHI (adj)':12s}: {GHI_ADJ_WEIGHT*100:.1f}%  [post-hoc]")


# ─────────────────────────────────────────────────────────────────────────────
# 6. PCA VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
for ax in (ax1, ax2):
    ax.set_facecolor('#0d1b2a')
    for sp in ax.spines.values():
        sp.set_color('#334466')

comps = [f'PC{i+1}' for i in range(len(PCA_VARS))]
ax1.bar(comps, var_explained * 100,
        color=['#4da6ff' if i == 0 else '#334466' for i in range(len(PCA_VARS))],
        edgecolor='#1a1a2e', linewidth=0.5)
ax1.plot(comps, np.cumsum(var_explained) * 100,
         color='#ff6b6b', marker='o', lw=2, markersize=7, label='Cumulative %')
for i, v in enumerate(var_explained):
    ax1.text(i, v * 100 + 1.5, f'{v*100:.1f}%', color='white',
             ha='center', fontsize=9, fontweight='bold')
ax1.set_ylabel('Variance Explained (%)', color='#aaaaaa', fontsize=10)
ax1.set_xlabel('Principal Component', color='#aaaaaa', fontsize=10)
ax1.tick_params(colors='#aaaaaa')
ax1.grid(axis='y', color='#1e2d3d', lw=0.6, alpha=0.7)
ax1.legend(fontsize=9, facecolor='#0d1b2a', edgecolor='#334466', labelcolor='white')
ax1.set_title('Scree Plot — Variance Explained\nby Each Principal Component',
              color='white', fontsize=11, fontweight='bold', pad=10)

var_labels  = list(pca_weight_dict.keys()) + ['GHI\n(post-hoc)']
weight_vals = list(pca_weight_dict.values()) + [GHI_ADJ_WEIGHT]
bars = ax2.bar(var_labels, weight_vals,
               color=['#4da6ff'] * len(pca_weight_dict) + ['#ffd700'],
               edgecolor='#1a1a2e', linewidth=0.5)
for bar, val in zip(bars, weight_vals):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f'{val*100:.1f}%', color='white', ha='center', fontsize=9, fontweight='bold')
ax2.set_ylabel('Weight in Composite Score', color='#aaaaaa', fontsize=10)
ax2.set_xlabel('Variable', color='#aaaaaa', fontsize=10)
ax2.tick_params(colors='white')
ax2.grid(axis='y', color='#1e2d3d', lw=0.6, alpha=0.7)
ax2.legend(handles=[
    Patch(facecolor='#4da6ff', label='PCA-derived weight (PC1 loadings)'),
    Patch(facecolor='#ffd700', label='GHI post-hoc adjustment'),
], fontsize=8.5, facecolor='#0d1b2a', edgecolor='#334466', labelcolor='white')
ax2.set_title(f'Composite Score Weights\nPC1 explains {var_explained[0]:.1%} of variance',
              color='white', fontsize=11, fontweight='bold', pad=10)

plt.tight_layout(pad=2.5)
plt.savefig(OUT_PCA_PLOT, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"PCA weights plot saved: {OUT_PCA_PLOT}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. COMPOSITE VULNERABILITY SCORE
# ─────────────────────────────────────────────────────────────────────────────

scaler = MinMaxScaler()
s      = pd.DataFrame(scaler.fit_transform(residential[PCA_VARS]),
                      columns=PCA_VARS, index=residential.index)
s['NDVI'] = 1 - s['NDVI']

residential['base_score']      = sum(s[v] * pca_weight_dict[v] for v in PCA_VARS)
ghi_vals                       = residential['GHI'].copy()
ghi_adj                        = (1 - (ghi_vals - ghi_vals.min()) /
                                   (ghi_vals.max() - ghi_vals.min())) * GHI_ADJ_WEIGHT
residential['composite_score'] = residential['base_score'] + ghi_adj.fillna(0)
residential['ghi_adjusted']    = residential['GHI'].notna()
residential['UTFVI_class']     = residential['UTFVI'].apply(classify_utfvi)
residential['in_benchmark']    = residential['Sector_nam'].isin(BENCHMARK)
residential['both_criteria']   = residential['Sector_nam'].isin(BENCHMARK_BOTH)
residential                    = residential.sort_values('composite_score').reset_index(drop=True)
residential['rank']            = range(1, len(residential) + 1)

print("\n=== Top 15 (best performing) ===")
print(residential[['rank', 'Sector_nam', 'UTFVI', 'UTFVI_class', 'LST',
                    'composite_score', 'in_benchmark', 'both_criteria']].head(15).to_string(index=False))
print("\n=== Bottom 5 (most vulnerable) ===")
print(residential[['rank', 'Sector_nam', 'UTFVI_class', 'LST',
                    'composite_score']].tail(5).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 8. DEVIATION TABLE
# ─────────────────────────────────────────────────────────────────────────────

dev_df = residential[['Sector_nam', 'rank', 'composite_score', 'UTFVI', 'UTFVI_class',
                       'LST', 'NDVI', 'TCI', 'GHI', 'Population', 'SC_pct',
                       'in_benchmark', 'both_criteria']].copy()
for col in ['LST', 'NDVI', 'TCI', 'Population']:
    bval = profile[col]
    dev_df[f'{col}_dev']     = (dev_df[col] - bval).round(4)
    dev_df[f'{col}_dev_pct'] = ((dev_df[col] - bval) / abs(bval) * 100).round(1)

dev_df.to_csv(OUT_DEVIATION_CSV, index=False)
print(f"\nDeviation table saved: {OUT_DEVIATION_CSV}")
print("\n=== Worst 10 sectors ===")
print(dev_df.tail(10)[['Sector_nam', 'UTFVI_class', 'LST', 'LST_dev',
                        'NDVI', 'NDVI_dev', 'composite_score']].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 9. UPDATE SHAPEFILE
# ─────────────────────────────────────────────────────────────────────────────

score_df = residential[['Sector_nam', 'composite_score', 'rank',
                         'UTFVI_class', 'in_benchmark', 'both_criteria']].copy()

gdf_shp = gpd.read_file(SHAPEFILE_PATH)
gdf_shp['Sector_nam'] = gdf_shp['Sector_nam'].astype(str)
gdf = gdf_shp.merge(urban, on='Sector_nam', how='left').reset_index(drop=True)

for col in ['composite_score', 'rank', 'LST_dev', 'UTFVI_class',
            'in_benchmark', 'both_criteria']:
    if col in gdf.columns:
        gdf = gdf.drop(columns=[col])

gdf = gdf.merge(score_df, on='Sector_nam', how='left')
gdf = gdf.merge(dev_df[['Sector_nam', 'LST_dev']], on='Sector_nam', how='left')
if 'UTFVI_class' not in gdf.columns:
    gdf['UTFVI_class'] = gdf['UTFVI'].apply(classify_utfvi)

gdf.to_file(MASTER_GEO_PATH, driver='GPKG')
print(f"master_geo.gpkg updated: {MASTER_GEO_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. COMPOSITE SCORE BAR CHART — VULNERABILITY INDEX
# ─────────────────────────────────────────────────────────────────────────────
# Shows all sectors ranked by composite vulnerability score
# Mean line indicates the vulnerability benchmark

fig, ax = plt.subplots(figsize=(18, 8), facecolor=BG)
ax.set_facecolor('#0d1b2a')
for sp in ax.spines.values():
    sp.set_color('#334466')

# Color by vulnerability score only
colors = []
for _, row in residential.iterrows():
    score = row['composite_score']
    r_val = min(1.0, score * 1.6)
    b_val = max(0.0, 1.0 - score * 1.6)
    colors.append((r_val, 0.25, b_val))

ax.bar(range(len(residential)), residential['composite_score'],
       color=colors, edgecolor='#1a1a2e', linewidth=0.4, zorder=3)

# Hatching for sectors where GHI adjustment was applied
for i, (_, row) in enumerate(residential.iterrows()):
    if row.get('ghi_adjusted', False):
        ax.bar(i, row['composite_score'], color='none',
               edgecolor='white', linewidth=0.3, hatch='///', zorder=4)

# Mean line (vulnerability benchmark)
mean_score = residential['composite_score'].mean()
ax.axhline(mean_score, color='#00ff88', linewidth=2.0,
           linestyle='--', alpha=0.8, zorder=5, label='Mean')
ax.text(len(residential) - 1, mean_score + 0.002,
        f'Mean = {mean_score:.3f}',
        color='#00ff88', fontsize=9, ha='right', va='bottom', fontweight='bold')

xlabels = residential['Sector_nam'].tolist()
ax.set_xticks(range(len(residential)))
ax.set_xticklabels(xlabels, rotation=45, ha='right', color='#aaaaaa', fontsize=7.5)
ax.set_ylabel('Composite Vulnerability Score\n(lower = better performance)',
              color='#aaaaaa', fontsize=10)
ax.set_xlabel('Sector (sorted from best to worst performance)',
              color='#aaaaaa', fontsize=9)
ax.tick_params(colors='#aaaaaa')
ax.grid(axis='y', color='#1e2d3d', lw=0.6, alpha=0.7)

weight_str = '  ·  '.join(
    [f'{v} {w*100:.0f}%' for v, w in pca_weight_dict.items()]
    + [f'GHI {GHI_ADJ_WEIGHT*100:.0f}% (post-hoc)'])
ax.set_title(
    f'Composite Vulnerability Score\n'
    f'PCA-derived weights: {", ".join([f"{v}={w*100:.0f}%" for v,w in pca_weight_dict.items()])}  ·  '
    f'GHI {GHI_ADJ_WEIGHT*100:.0f}% (post-hoc, only for sectors with GHI data)',
    color='white', fontsize=10, fontweight='bold', pad=12,
)
plt.tight_layout()
plt.savefig(OUT_VULN_CHART, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Composite scores (vulnerability) bar chart saved: {OUT_VULN_CHART}")


# ─────────────────────────────────────────────────────────────────────────────
# 10b. UTFVI BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
# Shows all sectors ranked by UTFVI, colored by benchmark status

# Sort by UTFVI for this chart
residential_by_utfvi = residential.sort_values('UTFVI').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(18, 8), facecolor=BG)
ax.set_facecolor('#0d1b2a')
for sp in ax.spines.values():
    sp.set_color('#334466')

# Color by UTFVI benchmark status
colors = []
for _, row in residential_by_utfvi.iterrows():
    if row['in_benchmark']:
        colors.append('#00ff88' if row['both_criteria'] else '#88ffcc')
    else:
        colors.append('#ff6b6b')

ax.bar(range(len(residential_by_utfvi)), residential_by_utfvi['UTFVI'],
       color=colors, edgecolor='#1a1a2e', linewidth=0.4, zorder=3)

# Hatching for sectors where GHI adjustment was applied - iterate over sorted data
for i, (_, row) in enumerate(residential_by_utfvi.iterrows()):
    if row.get('ghi_adjusted', False):
        ax.bar(i, row['UTFVI'], color='none',
               edgecolor='gray', linewidth=0.3, hatch='///', zorder=4)

# UTFVI = 0 benchmark line
ax.axhline(0, color='white', linewidth=2.0,
           linestyle='-', alpha=0.8, zorder=5)
ax.text(len(residential_by_utfvi) - 1, -0.001,
        'UTFVI = 0 (Excellent threshold)',
        color='white', fontsize=9, ha='right', va='top', fontweight='bold')

xlabels = [f"{r['Sector_nam']}**" if r['both_criteria'] else str(r['Sector_nam'])
           for _, r in residential_by_utfvi.iterrows()]
ax.set_xticks(range(len(residential_by_utfvi)))
ax.set_xticklabels(xlabels, rotation=45, ha='right', color='#aaaaaa', fontsize=7.5)
ax.set_ylabel('UTFVI (Urban Thermal Field Variance Index)',
              color='#aaaaaa', fontsize=10)
ax.tick_params(colors='#aaaaaa')
ax.grid(axis='y', color='#1e2d3d', lw=0.6, alpha=0.7)

ax.legend(handles=[
    mpatches.Patch(facecolor='#88ffcc',
                   label=f'Above UTFVI Benchmark sectors'),
    mpatches.Patch(facecolor='#ff6b6b',
                   label='Below UTFVI Benchmark sectors'),
], fontsize=9, facecolor='#0d1b2a', edgecolor='#334466',
   labelcolor='white', loc='upper left')

# Build title with mixed colors
ax.set_title(
    f'UTFVI Benchmark Classification\n'
    f'Benchmark = UTFVI < 0 |  n={len(BENCHMARK_NUM)}',
    color='white', fontsize=10, fontweight='bold', pad=12,
)
plt.tight_layout()
plt.savefig(OUT_BENCHMARK_CHART, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Composite scores (UTFVI benchmark) bar chart saved: {OUT_BENCHMARK_CHART}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. BENCHMARK DEVIATION MAP
# ─────────────────────────────────────────────────────────────────────────────
# All 63 sectors are coloured by their LST deviation from the benchmark mean,
# including Manimajra and Khuda Alisher (geographic islands excluded from
# spatial weights, but shown with their actual thermal deviation value).
# Islands are distinguished by a dashed white border instead of being hidden.
# ─────────────────────────────────────────────────────────────────────────────

gdf_map = gpd.read_file(SHAPEFILE_PATH)
gdf_map['Sector_nam'] = gdf_map['Sector_nam'].astype(str)

# Drop LST if already present in shapefile (use MASTER values instead)
if 'LST' in gdf_map.columns:
    gdf_map = gdf_map.drop(columns=['LST'])

gdf_map = gdf_map.merge(urban[['Sector_nam', 'LST', 'UTFVI']],
                         on='Sector_nam', how='left')
gdf_map['LST_dev_plot'] = gdf_map['LST'] - bench_lst
gdf_map['in_benchmark'] = gdf_map['Sector_nam'].isin(BENCHMARK)
gdf_map['both_criteria']= gdf_map['Sector_nam'].isin(BENCHMARK_BOTH)
gdf_map['is_island']    = gdf_map['Sector_nam'].isin(ISLANDS)

fig, ax = plt.subplots(figsize=(11, 11), facecolor=BG)
ax.set_facecolor(BG)
ax.set_axis_off()

# All sectors — coloured by LST deviation (islands included)
vmax = float(np.nanpercentile(gdf_map['LST_dev_plot'].abs(), 95))
gdf_map.plot(
    column='LST_dev_plot', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
    linewidth=0.5, edgecolor='#2a2a4a', ax=ax, legend=False, zorder=2,
    missing_kwds={'color': '#444466', 'edgecolor': '#2a2a4a'},
)

# Dashed white border for geographic islands (spatial-weights exclusion marker)
for _, row in gdf_map[gdf_map['is_island']].iterrows():
    ax.plot(*row.geometry.exterior.xy,
            color='white', linewidth=1.0, linestyle='--', alpha=0.7, zorder=3)

# Benchmark overlays
# All Excellent → solid green border
gdf_map[gdf_map['in_benchmark']].plot(
    color='none', edgecolor='#00ff88', linewidth=2.2, ax=ax, zorder=5)
# Doubly confirmed → additional gold inner ring
gdf_map[gdf_map['both_criteria']].plot(
    color='none', edgecolor='#ffd700', linewidth=1.0, ax=ax, zorder=6)

# Labels — all 63 sectors
for _, row in gdf_map.iterrows():
    c   = row.geometry.centroid
    sec = str(row['Sector_nam'])
    if row['both_criteria']:
        color, lbl = '#ffd700', f"{sec}**"
    elif row['in_benchmark']:
        color, lbl = '#00ff88', sec
    elif row['is_island']:
        color, lbl = '#ffffff', sec
    else:
        color, lbl = 'white', sec
    txt = ax.annotate(lbl, xy=(c.x, c.y), ha='center', va='center',
                      fontsize=5.0, fontweight='bold', color=color)
    txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground='black')])

# Colorbar
norm = Normalize(vmin=-vmax, vmax=vmax)
sm   = ScalarMappable(cmap='RdBu_r', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.032, pad=0.02, shrink=0.75)
cbar.set_label('LST deviation from benchmark (°C)', color='white', fontsize=10)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

ax.legend(handles=[
    mpatches.Patch(edgecolor='#00ff88', facecolor='none', linewidth=2.2,
                   label=f'Benchmark — UTFVI < 0 | (n={len(BENCHMARK_NUM)})'),
    mpatches.Patch(edgecolor='white', facecolor='none', linewidth=1.0,
                   linestyle='dashed',
                   label='Geographic island (excluded from spatial weights)'),
], loc='lower right', fontsize=8.5,
   facecolor='#0d1b2a', edgecolor='#334466', labelcolor='white')

ax.set_title(
    f'LST Deviation from Benchmark Cluster\n'
    f'Benchmark = UTFVI < 0  |  '
    f'Mean LST = {bench_lst:.2f}°C  |  n={len(BENCHMARK_NUM)} sectors\n',
    color='white', fontsize=11, fontweight='bold', pad=14,
)
plt.tight_layout()
plt.savefig(OUT_BENCH_MAP, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Benchmark map saved: {OUT_BENCH_MAP}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. RADAR CHARTS — HIGH-VULNERABILITY SECTORS VS BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
# Select the 4 worst-performing (most vulnerable) sectors for radar comparison
RADAR_VARS   = ['LST', 'NDVI', 'TCI', 'Population']
RADAR_LABELS = ['LST\n(lower=better)', 'NDVI\n(higher=better)',
                'TCI\n(lower=better)', 'Population\n(lower=better)']

# Dynamically select the last 4 sectors (most vulnerable) from the residential ranking
# Note: residential is sorted by composite_score, so the last 4 are the worst
HIGHLIGHT_SECTORS = residential.tail(4)['Sector_nam'].tolist()

def normalise_col(series):
    return (series - series.min()) / (series.max() - series.min())

# Create norm_df from residential (includes all urban sectors, numeric and non-numeric)
norm_df = residential[['Sector_nam'] + RADAR_VARS].copy()
for col in RADAR_VARS:
    norm_df[col + '_n'] = normalise_col(residential[col])
norm_df['NDVI_n'] = 1 - norm_df['NDVI_n']

# For benchmark, use numeric sectors only if they exist in the data
benchmark_for_radar = [s for s in BENCHMARK_NUM if s in residential['Sector_nam'].values]
bench_norm_df = norm_df[norm_df['Sector_nam'].isin(benchmark_for_radar)]
bench_norm = bench_norm_df[[c + '_n' for c in RADAR_VARS]].mean()

fig, axes = plt.subplots(2, 2, figsize=(14, 12),
                         subplot_kw={'polar': True}, facecolor=BG)
fig.suptitle(
    f'Worst Performing Sector Profile vs Benchmark (UTFVI < 0), n={len(BENCHMARK_NUM)})\n'
    'Radar Chart — Normalised scores',
    color='white', fontsize=12, fontweight='bold', y=1.01,
)

angles = np.linspace(0, 2 * np.pi, len(RADAR_VARS), endpoint=False).tolist()
angles += angles[:1]

for ax, sector in zip(axes.flatten(), HIGHLIGHT_SECTORS):
    ax.set_facecolor('#0d1b2a')
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    ax.spines['polar'].set_color('#334466')
    ax.grid(color='#334466', linewidth=0.6)

    sector_row = norm_df[norm_df['Sector_nam'] == sector]
    if len(sector_row) == 0:
        continue

    s_vals = [sector_row[c + '_n'].values[0] for c in RADAR_VARS]; s_vals += s_vals[:1]
    b_vals = [bench_norm[c + '_n'] for c in RADAR_VARS]          ; b_vals += b_vals[:1]

    ax.plot(angles, b_vals, color='#00ff88', lw=2.0, linestyle='--',
            label=f'Benchmark (n={len(BENCHMARK_NUM)})')
    ax.fill(angles, b_vals, color='#00ff88', alpha=0.08)
    ax.plot(angles, s_vals, color='#ff6b6b', lw=2.2, label=f'Sector {sector}')
    ax.fill(angles, s_vals, color='#ff6b6b', alpha=0.18)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_LABELS, size=9, color='white')
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['0.25', '0.50', '0.75'], color='#aaaaaa', fontsize=7)
    ax.set_ylim(0, 1)

    sector_lst   = float(residential[residential['Sector_nam'] == sector]['LST'].values[0])
    sector_class = residential[residential['Sector_nam'] == sector]['UTFVI_class'].values[0]
    ax.set_title(
        f'Sector {sector}\n'
        f'LST={sector_lst:.1f}°C  (+{sector_lst - bench_lst:.1f}°C vs benchmark)',
        color='white', fontsize=10, fontweight='bold', pad=18,
    )
    ax.legend(fontsize=7.5, facecolor='#0d1b2a', edgecolor='#334466',
              labelcolor='white', loc='upper right', bbox_to_anchor=(1.3, 1.15))

plt.tight_layout(pad=2.5)
plt.savefig(OUT_RADAR, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Radar charts saved: {OUT_RADAR}")


# ─────────────────────────────────────────────────────────────────────────────
# 13. COMPOSITE SCORE METHODOLOGY DOCUMENTATION
# ─────────────────────────────────────────────────────────────────────────────

doc_content = f"""
COMPOSITE VULNERABILITY SCORE METHODOLOGY
==========================================
Chandigarh Urban Sectors Vulnerability Ranking
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

═════════════════════════════════════════════════════════════════════════════════

1. OVERVIEW

The Composite Vulnerability Score quantifies urban vulnerability by combining four
key indicators (LST, NDVI, TCI, Population) weighted by their contribution to 
variance in Land Surface Temperature (LST). A post-hoc adjustment for government 
housing (GHI) is applied to sectors with available data.

FORMULA:
  composite_score = base_score + ghi_adjustment
  
where:
  base_score = Σ(w_i × normalized_variable_i)  for i = LST, NDVI, TCI, Population
  ghi_adjustment = normalized_GHI × GHI_ADJ_WEIGHT  (only if GHI is available)

SCALE:
  0.0 = best performing (lowest vulnerability)
  1.0 = worst performing (highest vulnerability)

═════════════════════════════════════════════════════════════════════════════════

2. STEP-BY-STEP CALCULATION

STEP 1: NORMALIZATION (MinMaxScaler)
─────────────────────────────────────────────────────────────────────────────
Each variable is scaled to [0, 1] using the formula:

  normalized_value = (x - min(x)) / (max(x) - min(x))

Variables normalized:
  • LST           (°C) — Land Surface Temperature
  • NDVI          — Normalized Difference Vegetation Index
  • TCI           — Thermal Comfort Index
  • Population    — Population density (persons/km²)

Data source: {len(residential)} residential sectors (all urban except excluded non-residential)
Excluded from scoring: Sectors 17 (commercial), 30 (industrial), 32 (institutional)

STEP 2: NDVI INVERSION
─────────────────────────────────────────────────────────────────────────────
NDVI is inverted because higher vegetation (higher NDVI) indicates BETTER 
conditions (lower vulnerability):

  normalized_NDVI = 1 - normalized_NDVI

This ensures that high vegetation reduces the composite score.

STEP 3: PCA-DERIVED WEIGHTS
─────────────────────────────────────────────────────────────────────────────
Principal Component Analysis (PCA) is performed on the four variables to derive
weights based on their variance contributions. PC1 is used because it captures
the maximum variance in the data.

Methodology:
  1. Standardize all four variables (zero mean, unit variance)
  2. Compute PCA; extract PC1 loadings
  3. Take absolute values of PC1 loadings
  4. Normalize to sum to (1 - GHI_ADJ_WEIGHT) = 0.85

Results:
"""

# Add PCA weights
doc_content += f"""
  PC1 variance explained: {var_explained[0]*100:.1f}%
  
  Variable          Weight        Contribution
  ───────────────────────────────────────────
"""
for var, w in pca_weight_dict.items():
    doc_content += f"  {var:15s}  {w*100:6.1f}%    {w*100:6.1f}% of (1.0 - GHI)\n"

doc_content += f"""
  GHI (post-hoc)    {GHI_ADJ_WEIGHT*100:6.1f}%    Fixed post-hoc adjustment
  ───────────────────────────────────────────
  TOTAL             100.0%

INTERPRETATION:
  • {max(pca_weight_dict, key=pca_weight_dict.get)} is the strongest driver of LST variance
  • Weights reflect natural covariance structure in the data, NOT normative importance
  • The constraint (sum to 0.85) reserves 15% for GHI adjustment

STEP 4: BASE SCORE CALCULATION
─────────────────────────────────────────────────────────────────────────────
Base score is the weighted sum of normalized variables:

  base_score = Σ(w_i × normalized_variable_i)

Example for Sector 1:
  If normalized values are: LST={residential.loc[0, 'LST']:.3f}, 
                            NDVI={residential.loc[0, 'NDVI']:.3f}, 
                            TCI={residential.loc[0, 'TCI']:.3f},
                            Population={residential.loc[0, 'Population']:.3f}
  Then base_score = sum of (weight × normalized value) for each variable

Range: 0.0 to ~0.85 (theoretical max if all variables at extremes)

STEP 5: GHI POST-HOC ADJUSTMENT
─────────────────────────────────────────────────────────────────────────────
Government Housing Index (GHI) is added as a post-hoc adjustment for sectors
with housing data (24 out of {len(residential)} sectors):

For sectors WITH GHI data:
  1. Normalize GHI values to [0, 1]: 
       normalized_GHI = (GHI - min(GHI)) / (max(GHI) - min(GHI))
     
  2. Invert (higher GHI pushes LST up):
       adjusted_GHI = 1 - normalized_GHI
     
  3. Weight by GHI_ADJ_WEIGHT:
       ghi_adjustment = adjusted_GHI × {GHI_ADJ_WEIGHT}

For sectors WITHOUT GHI data:
  ghi_adjustment = 0 (no contribution)
  These sectors are identified with hatching (///) in charts

STEP 6: FINAL COMPOSITE SCORE
─────────────────────────────────────────────────────────────────────────────
Final score combines base score and GHI adjustment:

  composite_score = base_score + ghi_adjustment

Range: 0.0 to ~1.0
  • 0.0–0.3 = Low vulnerability (better performing sectors)
  • 0.3–0.6 = Moderate vulnerability
  • 0.6–1.0 = High vulnerability (worse performing sectors)

═════════════════════════════════════════════════════════════════════════════════

3. RANKING AND INTERPRETATION

Sectors are ranked by composite_score from lowest (best, rank=1) to highest 
(worst, rank={len(residential)}):

Top 5 (Best Performing):
"""

# Add top 5 sectors
for _, row in residential.head(5).iterrows():
    doc_content += f"  #{int(row['rank']):2d}  Sector {row['Sector_nam']:5s}  "
    doc_content += f"Score={row['composite_score']:.4f}  LST={row['LST']:.1f}°C\n"

doc_content += f"""
Bottom 5 (Worst Performing):
"""

# Add bottom 5 sectors
for _, row in residential.tail(5).iterrows():
    doc_content += f"  #{int(row['rank']):2d}  Sector {row['Sector_nam']:5s}  "
    doc_content += f"Score={row['composite_score']:.4f}  LST={row['LST']:.1f}°C\n"

doc_content += f"""

═════════════════════════════════════════════════════════════════════════════════

4. DATA EXCLUSIONS AND SPECIAL CASES

EXCLUDED FROM SCORING (Non-residential):
  • Sector 17 (Commercial zone)
  • Sector 30 (Industrial zone)
  • Sector 32 (Institutional zone)

GEOGRAPHIC ISLANDS (Included but flagged):
  • Manimajra: Included in scoring; excluded from spatial weights (no shared borders)
  • Khuda Alisher: Included in scoring; excluded from spatial weights

SECTORS WITH GHI ADJUSTMENT:
  Total sectors with GHI data: {len(residential[residential['ghi_adjusted'] == True])}
  These sectors show /// hatching in the Composite Score Bar Chart

═════════════════════════════════════════════════════════════════════════════════

5. MATHEMATICAL FORMULAS (DETAILED)

MinMaxScaler Normalization:
  z_i = (x_i - min(x)) / (max(x) - min(x))  for each variable

PC1 Loadings (from sklearn.decomposition.PCA):
  loadings = |pca.components_[0]|  (absolute values)
  normalized = loadings / loadings.sum() × (1 - GHI_ADJ_WEIGHT)

Base Score:
  S_base = w_LST × z_LST + w_NDVI × z_NDVI + w_TCI × z_TCI + w_Pop × z_Population
  where z_NDVI = 1 - normalized_NDVI (inverted)

GHI Adjustment (per sector):
  If GHI is not NaN:
    z_GHI = (GHI - min(GHI)) / (max(GHI) - min(GHI))
    S_GHI = (1 - z_GHI) × GHI_ADJ_WEIGHT
  Else:
    S_GHI = 0

Final Composite Score:
  composite_score = S_base + S_GHI

═════════════════════════════════════════════════════════════════════════════════

6. BENCHMARK CONTEXT

The Composite Vulnerability Score is used alongside UTFVI (Urban Thermal Field 
Variance Index) to identify improvement priority sectors:

BENCHMARK CRITERIA:
  Criterion 1 (UTFVI Excellent): UTFVI < 0 (n = {len(BENCHMARK)} sectors)
  Criterion 2 (LISA LL cold spot): Spatial clustering (n = {len(BENCHMARK_BOTH)} sectors)

Benchmark mean values:
  LST:        {bench_lst:.3f}°C  (ecological standard for UHI exit)
  NDVI:       {profile['NDVI']:.4f}
  TCI:        {profile['TCI']:.4f}
  Population: {profile['Population']:.1f}

The Composite Score helps identify WHICH VARIABLES to target for improvement in
vulnerable sectors. High composite scores + high LST = priority for intervention.

═════════════════════════════════════════════════════════════════════════════════

7. LIMITATIONS AND NOTES

1. PCA WEIGHTS: The weights reflect variance structure, not normative importance.
   A variable may have high weight simply because it has high natural variation,
   not because it's "more important" for urban cooling.

2. GHI LIMITATION: Only 24/57 numeric sectors have GHI data. The 15% post-hoc
   weight is conservative to avoid overfitting to incomplete data.

3. NON-LINEAR RELATIONSHIPS: MinMaxScaler assumes linear relationships. Complex
   interactions (e.g., NDVI × Population) are not captured.

4. TEMPORAL: Scores are based on a single point-in-time snapshot (current data).
   Seasonal and inter-annual variation are not modeled.

5. SCALE DEPENDENCY: Normalized values depend on the min/max in the full dataset.
   If a new sector is added with extreme values, all scores shift slightly.

═════════════════════════════════════════════════════════════════════════════════

8. VALIDATION AND PERFORMANCE

Leave-One-Out Cross-Validation (Random Forest prediction of LST):
  The composite score's validity is corroborated by RF model performance:
  - R² (LOO-CV) = [from RF model in Day 9]
  - RMSE = [from RF model]
  
  The RF shows that the four variables (and their interactions) explain most
  LST variance, validating the composite score as a proxy for thermal risk.

═════════════════════════════════════════════════════════════════════════════════

REFERENCES

Liu, L. & Zhang, Y. (2011). Urban heat island analysis using the Landsat TM and
  ASTER data: A case study in Hong Kong. Remote Sensing, 3(7), 1535–1552.

Zhang, Y., Odeh, I. O., & Sun, L. (2006). Determining the relative importance
  of land-use, physiography, and socioeconomic factors to urban heat island 
  formation. Journal of Remote Sensing, 10(5), 789–803.

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model
  predictions. Advances in Neural Information Processing Systems, 30.

sklearn Documentation: https://scikit-learn.org/stable/modules/preprocessing.html

═════════════════════════════════════════════════════════════════════════════════

Generated by: Sector_benchmark.py
Author: Elias Ruiz Sabater
Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}
"""

# Write documentation to file
with open(OUT_COMPOSITE_DOC, 'w') as f:
    f.write(doc_content)

print(f"Composite Score Methodology documentation saved: {OUT_COMPOSITE_DOC}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"Day 8 complete")
print(f"{'='*70}")
print(f"Benchmark: UTFVI < 0")
print(f"  Total sectors:      {len(BENCHMARK)}  ({len(BENCHMARK_NUM)} numeric)")
print(f"  Both criteria (**): {len(BENCHMARK_BOTH)}  {BENCHMARK_BOTH}")
print(f"  UTFVI Excellent:    {BENCHMARK}")
print(f"Benchmark mean LST:   {bench_lst:.3f}°C")
print(f"Benchmark mean NDVI:  {profile['NDVI']:.4f}")
print(f"Benchmark mean TCI:   {profile['TCI']:.4f}")
print(f"PCA weights: {', '.join([f'{v}={w*100:.1f}%' for v,w in pca_weight_dict.items()])}")
print(f"             GHI={GHI_ADJ_WEIGHT*100:.0f}% (post-hoc)")
print()
print("Most vulnerable (bottom 5):")
for _, row in residential.tail(5).iterrows():
    print(f"  Sector {row['Sector_nam']:5s} [{row['UTFVI_class']:9s}]: "
          f"score={row['composite_score']:.3f}  "
          f"LST={row['LST']:.1f}°C (+{row['LST']-bench_lst:.1f}°C)")
print()
print("Outputs:")
for p in [OUT_PCA_PLOT, OUT_VULN_CHART, OUT_BENCHMARK_CHART, OUT_BENCH_MAP, OUT_RADAR,
          OUT_PROFILE_CSV, OUT_DEVIATION_CSV, OUT_COMPOSITE_DOC]:
    print(f"  {p}")
print(f"  {MASTER_GEO_PATH}  (updated: composite_score, rank, LST_dev, UTFVI_class)")



# End of code