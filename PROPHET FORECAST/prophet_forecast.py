"""
Chandigarh Multi-Layer Map Project
===================================
Prophet LST / NDVI / NDBI Forecasting — All Sectors
Produces:
  - prophet_forecasts.csv   (all sectors, 1990–2027, LST + NDVI + NDBI + UTFVI)
  - prophet_forecasts.png   (8 representative sector plots)

Author : Elias Ruiz Sabater
Date   : March 2026
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — adjust paths as needed
# ─────────────────────────────────────────────────────────────────────────────

URBAN_RURAL_PATH  = '/Users/eliasruizsabater/Desktop/Project MMU/Environmental Data/DATA-EXCEL/UrbanRural1990-2025.xlsx'   # GEE export
OUTPUT_CSV        = '/Users/eliasruizsabater/Desktop/Project MMU/RESULTS/PROPHET FORECAST/prophet_forecasts.csv'
OUTPUT_EVAL_CSV   = '/Users/eliasruizsabater/Desktop/Project MMU/RESULTS/PROPHET FORECAST/prophet_evaluation.csv'
OUTPUT_PLOT       = '/Users/eliasruizsabater/Desktop/Project MMU/RESULTS/PROPHET FORECAST/prophet_forecasts.png'

# Sectors to include in the representative plot
PLOT_SECTORS = ['17', '7', '10', '22', '45', 'Rural_All', 'Manimajra', '25']
PLOT_LABELS  = {
    '17'       : 'Sector 17 (Commercial hub)',
    '7'        : 'Sector 7 (Dense / mid-income)',
    '10'       : 'Sector 10 (High-income)',
    '22'       : 'Sector 22 (Inner city dense)',
    '45'       : 'Sector 45 (Outer residential)',
    'Rural_All': 'Rural All (Baseline)',
    'Manimajra': 'Manimajra (High density)',
    '25'       : 'Sector 25 (High SC%)',
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_excel(URBAN_RURAL_PATH)

df = df[['Sector_nam', 'Year', 'Satellite', 'LST', 'NDVI', 'NDBI', 'UTFVI', 'CityMeanLST']]
df['Sector_nam'] = df['Sector_nam'].astype(str)
df = df.dropna(subset=['Year', 'Sector_nam', 'LST'])
df['Year'] = df['Year'].astype(int)



# Where L5 and L7 overlap in the same year (e.g. 2000), take the mean
df_clean = df.groupby(['Sector_nam', 'Year'], as_index=False).agg({
    'LST'        : 'mean',
    'NDVI'       : 'mean',
    'NDBI'       : 'mean',
    'Satellite'  : 'first',
    'CityMeanLST': 'mean',
})

all_sectors = sorted(df_clean['Sector_nam'].unique())
print(f"Sectors to model: {len(all_sectors)}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. FIT PROPHET PER SECTOR — LST, NDVI, NDBI
# ─────────────────────────────────────────────────────────────────────────────
# Notes:
#   - yearly_seasonality=False  →  data is already an annual summer composite
#                                   (April–June); no sub-annual cycle to model
#   - changepoint_prior_scale   →  0.05 = conservative, prevents overfitting
#                                   on a 33-year series
#   - periods=3 with freq='YE'  →  from last observed year (2025) gives
#                                   2026 and 2027 as forecast years

results = []
eval_results = []

for sector in all_sectors:
    s = (df_clean[df_clean['Sector_nam'] == sector]
         [['Year', 'LST', 'NDVI', 'NDBI']]
         .sort_values('Year')
         .drop_duplicates('Year')
         .copy())

    # Skip sectors with too few data points to fit reliably
    if len(s) < 8:
        print(f"  Skipping {sector} — only {len(s)} data points")
        continue

    # Convert year to mid-summer datetime for Prophet
    s['ds'] = pd.to_datetime(s['Year'].astype(str) + '-06-15')

    # Fit a separate Prophet model for each variable
    fitted = {}
    for var in ['LST', 'NDVI', 'NDBI']:
        # Check if this variable has enough non-NaN values to fit
        var_data = s.rename(columns={var: 'y'})[['ds', 'y']].dropna()
        if len(var_data) < 2:
            print(f"  Skipping {sector} — {var} has only {len(var_data)} non-NaN values")
            fitted[var] = None
            continue
        
        m = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_mode='additive',
        )
        m.fit(var_data, algorithm='LBFGS')

        # Build future dataframe: 3 periods gets us end-2025, end-2026, end-2027
        future = m.make_future_dataframe(periods=3, freq='YE')
        # Normalise all dates to mid-summer (June 15) to match training data
        future['ds'] = future['ds'].apply(
            lambda d: pd.Timestamp(str(d.year) + '-06-15')
        )
        fitted[var] = m.predict(future)

    # Collect results for all years (historical + forecast)
    # Only proceed if LST model was successfully fitted
    if fitted['LST'] is None:
        continue
    for _, row in fitted['LST'].iterrows():
        year = row['ds'].year
        if year > 2027:
            continue

        obs_lst = s[s['Year'] == year]['LST'].values
        obs_ndvi = s[s['Year'] == year]['NDVI'].values
        obs_ndbi = s[s['Year'] == year]['NDBI'].values

# The above code is appending a dictionary to a list called `results`. The dictionary contains
# key-value pairs for various data related to a sector and year. Here's a breakdown of the key-value
# pairs in the dictionary:
        results.append({
            'Sector_nam'    : sector,
            'Year'          : year,
            'LST_observed'  : obs_lst[0] if len(obs_lst) > 0 else np.nan,
            'LST_forecast'  : round(row['yhat'],        4),
            'LST_lower'     : round(row['yhat_lower'],  4),
            'LST_upper'     : round(row['yhat_upper'],  4),
            'NDVI_observed' : obs_ndvi[0] if len(obs_ndvi) > 0 else np.nan,
            'NDVI_forecast': (
                round(
                    fitted['NDVI'].loc[fitted['NDVI']['ds'].dt.year == year, 'yhat'].values[0],
                    4
                )
                if fitted['NDVI'] is not None and len(fitted['NDVI'].loc[fitted['NDVI']['ds'].dt.year == year, 'yhat'].values) > 0
                else np.nan
            ),
            'NDBI_observed' : obs_ndbi[0] if len(obs_ndbi) > 0 else np.nan,
            'NDBI_forecast': (
                round(
                    fitted['NDBI'].loc[fitted['NDBI']['ds'].dt.year == year, 'yhat'].values[0],
                    4
                )
                if fitted['NDBI'] is not None and len(fitted['NDBI'].loc[fitted['NDBI']['ds'].dt.year == year, 'yhat'].values) > 0
                else np.nan
            ),
            'is_forecast'   : year > 2025,
        })

fc_df = pd.DataFrame(results)
print(f"Forecast rows: {len(fc_df)}  |  Sectors modelled: {fc_df['Sector_nam'].nunique()}")


# ─────────────────────────────────────────────────────────────────────────────
# 2B. CALCULATE EVALUATION METRICS (MAPE, MSE, MAE)
# ─────────────────────────────────────────────────────────────────────────────
# Only evaluate on historical data (where we have observations)

for sector in fc_df['Sector_nam'].unique():
    sector_data = fc_df[(fc_df['Sector_nam'] == sector) & (fc_df['is_forecast'] == False)].copy()
    
    if len(sector_data) < 2:
        continue
    
    # Evaluate each variable where we have both observed and forecast values
    for var in ['LST', 'NDVI', 'NDBI']:
        obs_col = f'{var}_observed'
        fcst_col = f'{var}_forecast'
        
        # Filter to rows with both observed and forecast values
        valid_data = sector_data.dropna(subset=[obs_col, fcst_col])
        
        if len(valid_data) < 2:
            continue
        
        y_true = valid_data[obs_col].values
        y_pred = valid_data[fcst_col].values
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero by filtering out zero values
        if np.any(y_true != 0):
            mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, np.nan))) * 100
        else:
            mape = np.nan
        
        eval_results.append({
            'Sector_nam': sector,
            'Variable': var,
            'N_observations': len(valid_data),
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'MSE': round(mse, 4),
            'MAPE_%': round(mape, 2) if not np.isnan(mape) else np.nan,
        })

eval_df = pd.DataFrame(eval_results)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DERIVE UTFVI FORECAST
# ─────────────────────────────────────────────────────────────────────────────
# UTFVI = (LST_sector - CityMeanLST) / CityMeanLST
# CityMeanLST is computed from urban numeric sectors only (consistent with GEE script)
# Note: GEE script uses denominator = LST_sector (not CityMeanLST).
#       This matches the formula in the exported dataset.
#       The Elsevier paper uses denominator = CityMeanLST. Document as limitation.

urban_mask = fc_df['Sector_nam'].str.match(r'^\d+$')
city_mean  = (fc_df[urban_mask]
              .groupby('Year')['LST_forecast']
              .mean()
              .rename('CityMean_forecast'))

fc_df = fc_df.merge(city_mean, on='Year', how='left')
fc_df['UTFVI_forecast'] = (
    (fc_df['LST_forecast'] - fc_df['CityMean_forecast']) / fc_df['CityMean_forecast']
).round(4)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVE CSV
# ─────────────────────────────────────────────────────────────────────────────

# Deduplicate 2025: keep the observed row (is_forecast=False), drop the extra
fc_df = fc_df.sort_values(['Sector_nam', 'Year', 'is_forecast'])
fc_df = fc_df.drop_duplicates(subset=['Sector_nam', 'Year'], keep='first')

fc_df.to_csv(OUTPUT_CSV, index=False, sep=',', decimal='.')
print(f"Saved: {OUTPUT_CSV}")

# Save evaluation metrics
if len(eval_df) > 0:
    # Add average metrics for each variable across all sectors
    avg_rows = []
    for var in ['LST', 'NDVI', 'NDBI']:
        var_data = eval_df[eval_df['Variable'] == var]
        if len(var_data) > 0:
            avg_row = {
                'Sector_nam': 'AVERAGE_ALL',
                'Variable': var,
                'N_observations': var_data['N_observations'].sum(),
                'MAE': round(var_data['MAE'].mean(), 4),
                'RMSE': round(var_data['RMSE'].mean(), 4),
                'MSE': round(var_data['MSE'].mean(), 4),
                'MAPE_%': round(var_data['MAPE_%'].mean(), 2),
            }
            avg_rows.append(avg_row)
    
    eval_df_with_avg = pd.concat([eval_df, pd.DataFrame(avg_rows)], ignore_index=True)
    eval_df_with_avg.to_csv(OUTPUT_EVAL_CSV, index=False, sep=',', decimal='.')
    print(f"Saved: {OUTPUT_EVAL_CSV}")
    print("\nEvaluation Metrics Summary:")
    print(eval_df_with_avg.to_string(index=False))
else:
    print("No evaluation metrics to save (insufficient historical data)")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PRINT SUMMARY STATS
# ─────────────────────────────────────────────────────────────────────────────

future = fc_df[fc_df['is_forecast'] == True]

urban_2026 = future[(future['Year'] == 2026) & urban_mask]['LST_forecast'].mean()
urban_2027 = future[(future['Year'] == 2027) & urban_mask]['LST_forecast'].mean()
rural_2026 = future[(future['Year'] == 2026) & (fc_df['Sector_nam'] == 'Rural_All')]['LST_forecast'].values
rural_2027 = future[(future['Year'] == 2027) & (fc_df['Sector_nam'] == 'Rural_All')]['LST_forecast'].values

print(f"\nUrban mean LST:  2026 = {urban_2026:.2f}°C  |  2027 = {urban_2027:.2f}°C")
if len(rural_2026):
    print(f"Rural_All LST:   2026 = {rural_2026[0]:.2f}°C  |  2027 = {rural_2027[0]:.2f}°C")
    print(f"UHI gap:         2026 = {urban_2026 - rural_2026[0]:.2f}°C  "
          f"|  2027 = {urban_2027 - rural_2027[0]:.2f}°C")

print("\nTop 5 hottest urban sectors in 2027:")
print(future[(future['Year'] == 2027) & urban_mask]
      .nlargest(5, 'LST_forecast')[['Sector_nam', 'LST_forecast', 'LST_lower', 'LST_upper']]
      .to_string(index=False))

print("\nTop 5 coolest urban sectors in 2027:")
print(future[(future['Year'] == 2027) & urban_mask]
      .nsmallest(5, 'LST_forecast')[['Sector_nam', 'LST_forecast', 'LST_lower', 'LST_upper']]
      .to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOT — 8 REPRESENTATIVE SECTORS
# ─────────────────────────────────────────────────────────────────────────────

BG = '#0d1117'
fig, axes = plt.subplots(2, 4, figsize=(22, 12), facecolor=BG)
fig.suptitle(
    'Prophet LST Forecasts — Representative Sectors\n'
    'Chandigarh Summer LST (°C)  1990–2027  |  Shaded band = 80% confidence interval',
    color='white', fontsize=13, fontweight='bold', y=1.01,
)

for idx, sector in enumerate(PLOT_SECTORS):
    ax = axes[idx // 4][idx % 4]
    ax.set_facecolor('#0d1b2a')
    for sp in ax.spines.values():
        sp.set_color('#334466')

    # Re-fit the LST model for this sector (for plotting)
    s = (df_clean[df_clean['Sector_nam'] == sector][['Year', 'LST']]
         .sort_values('Year').drop_duplicates('Year').copy())
    s['ds'] = pd.to_datetime(s['Year'].astype(str) + '-06-15')

    m = Prophet(
        yearly_seasonality=False, weekly_seasonality=False,
        daily_seasonality=False, changepoint_prior_scale=0.05,
    )
    # Re-fit Prophet model here to ensure the plot reflects the latest data and to avoid side effects from earlier fits
    m.fit(s.rename(columns={'LST': 'y'})[['ds', 'y']], algorithm='LBFGS')

    future_df = m.make_future_dataframe(periods=3, freq='YE')
    future_df['ds'] = future_df['ds'].apply(
        lambda d: pd.Timestamp(str(d.year) + '-06-15'))
    fc = m.predict(future_df)
    fc['year'] = fc['ds'].dt.year

    # Split into historical fitted trend and forecast
    hist_fc = fc[fc['year'] <= 2025].copy()   # ends at 2025
    fore_fc = fc[fc['year'] >= 2026].copy()   # starts at 2026 — no shared point

    # Round forecast values to 2 decimal places
    hist_fc['yhat'] = hist_fc['yhat'].round(2)
    hist_fc['yhat_lower'] = hist_fc['yhat_lower'].round(2)
    hist_fc['yhat_upper'] = hist_fc['yhat_upper'].round(2)
    
    fore_fc['yhat'] = fore_fc['yhat'].round(2)
    fore_fc['yhat_lower'] = fore_fc['yhat_lower'].round(2)
    fore_fc['yhat_upper'] = fore_fc['yhat_upper'].round(2)

    # ── Observed scatter ──────────────────────────────────────────────────
    ax.scatter(s['Year'], s['LST'],
               color='white', s=22, zorder=5, alpha=0.8, label='Observed')

    # ── Historical fitted trend (blue) ────────────────────────────────────
    ax.plot(hist_fc['year'], hist_fc['yhat'],
            color='#4da6ff', lw=1.8, zorder=3, label='Trend (fitted)')
    ax.fill_between(hist_fc['year'], hist_fc['yhat_lower'], hist_fc['yhat_upper'],
                    color='#4da6ff', alpha=0.10, zorder=2)

    # ── Forecast (red dashed) — starts at 2026, does NOT share point with trend
    ax.plot(fore_fc['year'], fore_fc['yhat'],
            color='#ff6b6b', lw=2.2, linestyle='--', zorder=4, label='Forecast')
    ax.fill_between(fore_fc['year'], fore_fc['yhat_lower'], fore_fc['yhat_upper'],
                    color='#ff6b6b', alpha=0.15, zorder=2)

    # ── Vertical divider at 2025/2026 boundary ────────────────────────────
    ax.axvline(2025.5, color='#88aacc', linewidth=1.2, linestyle='--',
               alpha=0.6, zorder=6)
    ylo, yhi = ax.get_ylim()
    ax.text(2025.6, ylo + (yhi - ylo) * 0.04,
            'Forecast ▶', color='#88aacc', fontsize=7, va='bottom')

    # ── Forecast value annotations ────────────────────────────────────────
    for i, (_, row) in enumerate(fore_fc.iterrows()):
        offset = -18 if i == 0 else 12      # 2026 below the point, 2027 above
        va     = 'top' if i == 0 else 'bottom'
        ax.annotate(
            f"{int(row['year'])}: {row['yhat']:.2f}°C",
            xy=(row['year'], row['yhat']),
            xytext=(0, offset), textcoords='offset points',
            color='#ff9999', fontsize=8, ha='center', fontweight='bold', va=va,
        )

    # ── Calculate and display trend steepness (slope) ──────────────────────
    # Slope = change in LST per year over the entire historical period
    if len(hist_fc) >= 2:
        hist_years = hist_fc['year'].values
        hist_values = hist_fc['yhat'].values
        slope = (hist_values[-1] - hist_values[0]) / (hist_years[-1] - hist_years[0])
        
        # Add slope annotation in the top-right corner
        ax.text(0.98, 0.98, f'Slope: {slope:+.3f}°C/year',
                transform=ax.transAxes, fontsize=7.5, color='#4da6ff',
                ha='right', va='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#0d1117',
                         edgecolor='#4da6ff', linewidth=0.8, alpha=0.8))

    # ── Axis formatting ───────────────────────────────────────────────────
    ax.set_title(PLOT_LABELS[sector], color='white', fontsize=9.5,
                 fontweight='bold', pad=6)
    ax.set_xlabel('Year',   color='#aaaaaa', fontsize=8)
    ax.set_ylabel('LST (°C)', color='#aaaaaa', fontsize=8)
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    ax.grid(axis='y', color='#1e2d3d', lw=0.6)

    if idx == 0:
        ax.legend(fontsize=7.5, facecolor='#0d1b2a', edgecolor='#334466',
                  labelcolor='white', loc='upper left')

plt.tight_layout(pad=2.5)
plt.savefig(OUTPUT_PLOT, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Saved: {OUTPUT_PLOT}")
