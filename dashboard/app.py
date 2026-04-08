"""
Chandigarh Multi-Layer Map — Streamlit Dashboard
=================================================
Assessment of Environmental, Socio-Economic and Public Health Issues
Through a Multi-Layer Map of Chandigarh City

Author  : Elias Ruiz Sabater
Supervisor : Dr. Rajiv Chechi  |  Institution: MMDU
Week 7 Day 10  |  April 2026

4 Panels:
  1. City Overview Map      — interactive choropleth (LST / NDVI / UTFVI / composite score)
  2. Sector Explorer        — sector selector → full profile + time-series forecast chart
  3. SHAP Improvement Guide — Tensioned intervention guidelines with priority ranking
  4. Benchmark Comparison   — all sectors vs benchmark radar + deviation table

Run locally:
    streamlit run app.py

Deploy:
    Push this folder to GitHub → connect to Streamlit Cloud (free tier)
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Chandigarh Multi-Layer Map",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #0d1117; color: #e0e0e0; }
  /* Sidebar */
  [data-testid="stSidebar"] { background-color: #0d1b2a; }
  /* Metric cards */
  [data-testid="metric-container"] {
      background: #1a2744; border-radius: 8px; padding: 12px;
      border: 1px solid #334466;
  }
  /* Tab labels */
  .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
  /* Section headers */
  h1, h2, h3 { color: #4da6ff; }
  /* Tables */
  .dataframe { background-color: #0d1b2a !important; }
  /* Info boxes */
  .stInfo { background-color: #0d1b2a; border-color: #334466; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING — cached for performance
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_data
def load_data():
    params   = pd.read_csv(os.path.join(DATA_DIR, "sector_parameter_table.csv"))
    guide    = pd.read_csv(os.path.join(DATA_DIR, "improvement_guidelines.csv"))
    forecast = pd.read_csv(os.path.join(DATA_DIR, "prophet_forecasts.csv"))
    dev      = pd.read_csv(os.path.join(DATA_DIR, "sector_benchmark_deviation.csv"))
    bench_p  = pd.read_csv(os.path.join(DATA_DIR, "benchmark_profile.csv"))

    # Standardise sector name column
    for df in [params, dev]:
        df['Sector_nam'] = df['Sector_nam'].astype(str)
    guide['Sector']   = guide['Sector'].astype(str)
    forecast['Sector_nam'] = forecast['Sector_nam'].astype(str)

    # Classify UTFVI
    def classify_utfvi(v):
        if pd.isna(v):   return 'N/A'
        if v < 0:        return 'Excellent'
        elif v < 0.005:  return 'Good'
        elif v < 0.010:  return 'Normal'
        elif v < 0.015:  return 'Bad'
        elif v < 0.020:  return 'Worse'
        else:            return 'Worst'

    params['UTFVI_class'] = params['UTFVI'].apply(classify_utfvi)

    # Benchmark profile as dict
    bench = bench_p.set_index('Variable')['Benchmark_mean'].to_dict()

    # GeoJSON
    geojson_path = os.path.join(DATA_DIR, "sectors_wgs84.geojson")
    with open(geojson_path) as f:
        geojson = json.load(f)

    return params, guide, forecast, dev, bench, geojson

params, guide, forecast, dev, bench, geojson = load_data()

# Useful lists
ALL_SECTORS     = sorted(params['Sector_nam'].tolist(),
                         key=lambda x: int(x) if x.isdigit() else 999)
GROUP_A_SECTORS = guide['Sector'].tolist()
BENCH_LST       = bench.get('LST', 39.13)
CITY_MEAN_LST   = float(params['LST'].mean())

# Lookup dict: sector → row
sector_lookup = params.set_index('Sector_nam').to_dict('index')

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🗺️ Chandigarh Multi-Layer Map")
    st.markdown("*Environmental and Socio-Economic Assessment*")
    st.markdown("---")

    st.markdown("**Project**")
    st.markdown("Assessment of Chandigarh Environmental and Socio-Economic situation through a Multi-Layer Map")
    st.markdown(f"**Author:** Elias Ruiz Sabater  \n**Supervisor:** Dr. Rajiv Chechi  \n**Institution:** MMDU")
    st.markdown("---")

    st.markdown("**Key figures**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Urban sectors", "63")
        st.metric("Benchmark LST", f"{BENCH_LST:.2f}°C")
    with col2:
        st.metric("City mean LST", f"{CITY_MEAN_LST:.2f}°C")
        st.metric("Tensioned sectors", str(len(GROUP_A_SECTORS)))
    st.markdown("---")

    st.markdown("**UTFVI Classes (Zhang 2006)**")
    utfvi_info = {
        "Excellent": ("< 0", "#2c7bb6"),
        "Good": ("0 – 0.005", "#74add1"),
        "Normal": ("0.005 – 0.010", "#ffffbf"),
        "Bad": ("0.010 – 0.015", "#fdae61"),
        "Worse": ("0.015 – 0.020", "#f46d43"),
        "Worst": ("> 0.020", "#d73027"),
    }
    for cls, (rng, color) in utfvi_info.items():
        st.markdown(
            f'<span style="color:{color};font-weight:bold">■</span> '
            f'**{cls}** — UTFVI {rng}',
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️  City Overview Map",
    "🔍  Sector Explorer",
    "⚡  SHAP Improvement Guide",
    "📊  Benchmark Comparison",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — CITY OVERVIEW MAP
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("## City Overview")
    st.markdown(
        "Select a variable to colour the map. Click any| sector to see its values.<br>"
        "<span style='color:#2c7bb6'>Blue border</span> = UTFVI &lt; 0.0 (benchmark cluster with respect to mean city LST), n = 30.<br>"
        "<span style='color:#ff6b6b'>Red border</span> = Tensioned sectors. Below UTFVI and Multi-variable Composite Score benchmarks, n = 20.",
        unsafe_allow_html=True
    )

    col_ctrl, col_map = st.columns([1, 3])

    with col_ctrl:
        map_var = st.selectbox(
            "Colour sectors by:",
            ["LST (°C)", "NDVI", "UTFVI", "TCI (traffic index)", "Multi-variable Composite score"],
            index=0,
        )
        show_benchmark = st.checkbox("Highlight benchmark sectors", value=True)
        show_group_a   = st.checkbox("Highlight Tensioned sectors", value=False)

        st.markdown("---")
        st.markdown("**Variable ranges**")

        var_col_map = {
            "LST (°C)"      : ("LST",             "RdBu_r", False),
            "NDVI"          : ("NDVI",             "RdYlGn", False),
            "UTFVI"         : ("UTFVI",            "RdBu_r", False),
            "TCI (traffic index)" : ("TCI",              "Reds",   False),
            "Multi-variable Composite score": ("composite_score", "RdYlGn", True),
        }
        col_name, cmap_name, invert = var_col_map[map_var]

        # Use params or dev for composite_score
        if col_name == "composite_score":
            map_df = dev[['Sector_nam', 'composite_score']].copy()
            map_df.columns = ['Sector_nam', 'value']
        else:
            map_df = params[['Sector_nam', col_name]].copy()
            map_df.columns = ['Sector_nam', 'value']

        vmin = float(map_df['value'].min())
        vmax = float(map_df['value'].max())
        st.metric("Min", f"{vmin:.3f}")
        st.metric("Max", f"{vmax:.3f}")
        st.metric("City mean", f"{map_df['value'].mean():.3f}")

        # UTFVI distribution if showing UTFVI
        if col_name == "UTFVI":
            st.markdown("---")
            st.markdown("**Ecological class distribution**")
            dist = params['UTFVI_class'].value_counts()
            for cls in ["Excellent","Good","Normal","Bad","Worse","Worst"]:
                if cls in dist.index:
                    color = utfvi_info[cls][1]
                    st.markdown(
                        f'<span style="color:{color}">■</span> {cls}: **{dist[cls]}** sectors',
                        unsafe_allow_html=True
                    )

    with col_map:
        # Build Folium map
        m = folium.Map(
            location=[30.735, 76.788],
            zoom_start=12,
            tiles="CartoDB dark_matter",
        )

        # Build value dict for colour scale
        val_dict = map_df.set_index('Sector_nam')['value'].to_dict()

        # Colour scale helper
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        def get_color(val, vmin, vmax, cmap_name, invert=False):
            if pd.isna(val):
                return "#444466"
            norm = (val - vmin) / (vmax - vmin + 1e-9)
            if invert:
                norm = 1 - norm
            cmap = plt.get_cmap(cmap_name)
            rgba = cmap(norm)
            return mcolors.to_hex(rgba)

        benchmark_set = set(
            params[params['UTFVI'] < 0]['Sector_nam'].tolist()
        )

        # Add sector polygons
        for feature in geojson['features']:
            sec = str(feature['properties'].get('Sector_nam', ''))
            val = val_dict.get(sec, np.nan)
            fill_color = get_color(val, vmin, vmax, cmap_name, invert)

            is_bench   = sec in benchmark_set and show_benchmark
            is_group_a = sec in GROUP_A_SECTORS and show_group_a

            border_color = "#2c7bb6" if is_bench else ("#ff6b6b" if is_group_a else "#334466")
            border_weight= 2.5 if (is_bench or is_group_a) else 0.8

            tooltip_text = (
                f"<b>Sector {sec}</b><br>"
                f"{col_name}: {val:.3f}<br>"
            )

            # Only add UTFVI line if it's not the currently selected variable
            if col_name != "UTFVI":
                uv = sector_lookup.get(sec, {}).get('UTFVI', np.nan)
                if pd.isna(uv):
                    uv_str = "N/A"
                else:
                    uv_str = f"{uv:.4f}"
                tooltip_text += f"UTFVI: {uv_str}<br>"
            else:
                tooltip_text += f"Class: {sector_lookup.get(sec, {}).get('UTFVI_class','N/A')}"

            folium.GeoJson(
                feature,
                style_function=lambda f, fc=fill_color, bc=border_color, bw=border_weight: {
                    "fillColor"   : fc,
                    "fillOpacity" : 0.78,
                    "color"       : bc,
                    "weight"      : bw,
                },
                tooltip=folium.Tooltip(tooltip_text, sticky=True),
            ).add_to(m)

        st_folium(m, width=900, height=580, returned_objects=[])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — SECTOR EXPLORER
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("## Sector Explorer")
    st.markdown("Select a sector to see its full environmental profile, forecast trajectory and comparison against the benchmark.")

    col_sel, col_profile = st.columns([1, 3])

    with col_sel:
        selected_sector = st.selectbox(
            "Select sector:",
            ALL_SECTORS,
            index=ALL_SECTORS.index("45") if "45" in ALL_SECTORS else 0,
        )
        st.markdown("---")
        # Quick status indicators
        row = sector_lookup.get(selected_sector, {})
        utfvi_val = row.get("UTFVI", np.nan)
        utfvi_cls = row.get("UTFVI_class", "N/A")
        lst_val   = row.get("LST", np.nan)
        lst_gap   = lst_val - BENCH_LST if not pd.isna(lst_val) else np.nan

        st.markdown("**Quick status**")
        status_color = "#d73027" if utfvi_cls == "Worst" else (
            "#2c7bb6" if utfvi_cls == "Excellent" else "#fdae61")
        st.markdown(
            f'<div style="background:#1a2744;border-radius:8px;padding:12px;'
            f'border-left:4px solid {status_color}">'
            f'<b>UTFVI Class:</b> <span style="color:{status_color}">{utfvi_cls}</span><br>'
            f'<b>LST:</b> {lst_val:.2f}°C<br>'
            f'<b>Gap to benchmark:</b> {"+" if lst_gap>0 else ""}{lst_gap:.2f}°C<br>'
            f'<b>Tensioned sectors:</b> {"Yes 🔴" if selected_sector in GROUP_A_SECTORS else "No"}'
            f'</div>',
            unsafe_allow_html=True
        )

        if selected_sector in GROUP_A_SECTORS:
            st.markdown("---")
            g_row = guide[guide['Sector'] == selected_sector]
            if len(g_row):
                g = g_row.iloc[0]
                st.markdown("**Intervention guidelines**")
                st.markdown(f"**Priority rank:** #{int(g['Priority_rank'])}")
                st.markdown(f"**Required ΔNDVI:** +{g['ΔNDVI_required']:.3f}")
                if g['ΔTCI_required'] < 0:
                    st.markdown(f"**Required ΔTCI:** {g['ΔTCI_required']:.3f}")
                st.markdown(f"**Target LST:** {g['LST_target']:.2f}°C")
                st.markdown(f"**Target UTFVI class:** {g['UTFVI_class_tgt']}")
                st.markdown(f"**Intervention:** {g['Intervention_type']}")

    with col_profile:
        # ── Metrics row ──────────────────────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        # For LST: positive gap (hotter than benchmark) should be red, negative (cooler) should be green
        # Using delta_color parameter to inverse the color interpretation
        m1.metric("LST", f"{row.get('LST', 0):.2f}°C",
                  delta=f"{lst_gap:+.2f}°C vs benchmark" if not pd.isna(lst_gap) else "",
                  delta_color="inverse" if lst_gap != 0 else "off")
        m2.metric("NDVI", f"{row.get('NDVI', 0):.4f}")
        m3.metric("NDBI", f"{row.get('NDBI', 0):.4f}")
        m4.metric("TCI",  f"{row.get('TCI', 0):.3f}")
        m5.metric("UTFVI",f"{row.get('UTFVI', 0):.4f}")

        st.markdown("---")

        # ── Forecast time-series chart ───────────────────────────────────────
        st.markdown(f"#### LST Time-Series & Forecast — Sector {selected_sector}")

        fc_s = forecast[forecast['Sector_nam'] == selected_sector].copy()
        fc_s = fc_s.sort_values('Year')

        obs      = fc_s[fc_s['is_forecast'] == False]
        fcast    = fc_s[fc_s['is_forecast'] == True]

        fig_fc = go.Figure()

        # Confidence interval
        if len(fcast):
            fig_fc.add_trace(go.Scatter(
                x=list(fcast['Year']) + list(fcast['Year'])[::-1],
                y=list(fcast['LST_upper']) + list(fcast['LST_lower'])[::-1],
                fill='toself', fillcolor='rgba(77,166,255,0.15)',
                line=dict(color='rgba(0,0,0,0)'),
                name='95% CI', showlegend=True,
            ))

        # Historical observed
        fig_fc.add_trace(go.Scatter(
            x=obs['Year'], y=obs['LST_observed'],
            mode='markers+lines',
            marker=dict(size=5, color='#aaaaaa'),
            line=dict(color='#aaaaaa', width=1.5, dash='dot'),
            name='Observed LST',
        ))

        # Historical trend (Prophet fitted)
        fig_fc.add_trace(go.Scatter(
            x=fc_s[fc_s['is_forecast']==False]['Year'],
            y=fc_s[fc_s['is_forecast']==False]['LST_forecast'],
            mode='lines', line=dict(color='#4da6ff', width=2),
            name='Prophet trend (historical)',
        ))

        # Forecast
        if len(fcast):
            fig_fc.add_trace(go.Scatter(
                x=fcast['Year'], y=fcast['LST_forecast'],
                mode='lines+markers',
                marker=dict(size=8, color='#ff6b6b', symbol='diamond'),
                line=dict(color='#ff6b6b', width=2.5),
                name='LST Forecast (2026–2027)',
            ))

        # Benchmark line
        fig_fc.add_hline(y=BENCH_LST, line_color='#00ff88', line_dash='dash', line_width=1.5,
                         annotation_text=f"Benchmark {BENCH_LST:.2f}°C",
                         annotation_font_color='#00ff88')

        fig_fc.update_layout(
            template='plotly_dark', paper_bgcolor='#0d1b2a', plot_bgcolor='#0d1b2a',
            margin=dict(l=40,r=20,t=20,b=40), height=320,
            legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1),
            xaxis=dict(title='Year', gridcolor='#1e2d3d'),
            yaxis=dict(title='LST (°C)', gridcolor='#1e2d3d'),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # ── Radar chart: sector vs benchmark ────────────────────────────────
        st.markdown(f"#### Profile vs Benchmark — Sector {selected_sector}")
        st.caption(
            "**Green dashed** = UTFVI-Excellent benchmark (fixed reference — same for all sectors). "
            "**Grey dotted** = city mean (changes scale). "
            "**Red solid** = selected sector. "
            "Larger area = worse performance. NDVI is inverted (higher vegetation = better)."
        )
 
        radar_vars  = ['LST', 'NDVI', 'TCI', 'Population']
        radar_label = ['LST\n(lower=better)', 'NDVI\n(higher=better)',
                       'TCI\n(lower=better)', 'Population\n(lower=better)']
 
        # Normalise over all 63 sectors so named sectors render correctly
        norm_df = params.copy()
 
        def norm(series):
            mn, mx = series.min(), series.max()
            return (series - mn) / (mx - mn + 1e-9)
 
        for rv in radar_vars:
            norm_df[rv+'_n'] = norm(params[rv])
        norm_df['NDVI_n'] = 1 - norm_df['NDVI_n']   # invert: higher NDVI = lower norm
 
        # ── Three reference polygons ─────────────────────────────────────────
        # 1. Benchmark: mean of UTFVI-Excellent sectors (FIXED — same for all sectors)
        #    This is the ecological standard. It does not change per sector because
        #    a benchmark is a ruler, not a moving target.
        bench_mask = norm_df['UTFVI'] < 0
        b_vals = [norm_df.loc[bench_mask, rv+'_n'].mean() for rv in radar_vars]
 
        # 2. City mean: mean of all 63 sectors
        #    Gives the user a sense of where the "average" sector sits.
        #    This ALSO does not change per sector — it is a city-wide reference.
        c_vals = [norm_df[rv+'_n'].mean() for rv in radar_vars]
 
        # 3. Selected sector
        sec_row = norm_df[norm_df['Sector_nam'] == selected_sector]
        if len(sec_row):
            s_vals = [float(sec_row[rv+'_n'].values[0]) for rv in radar_vars]
        else:
            s_vals = [0.5] * len(radar_vars)
 
        # ── Per-variable gap table (what actually changes per sector) ─────────
        # Raw values for display
        raw_row = params[params['Sector_nam'] == selected_sector]
        bench_raw = {
            'LST'       : float(params.loc[bench_mask, 'LST'].mean()),
            'NDVI'      : float(params.loc[bench_mask, 'NDVI'].mean()),
            'TCI'       : float(params.loc[bench_mask, 'TCI'].mean()),
            'Population': float(params.loc[bench_mask, 'Population'].mean()),
        }
 
        fig_radar = go.Figure()
 
        # City mean polygon (bottom layer)
        fig_radar.add_trace(go.Scatterpolar(
            r=c_vals + [c_vals[0]], theta=radar_label + [radar_label[0]],
            fill='toself', fillcolor='rgba(150,150,150,0.06)',
            line=dict(color='#666688', width=1.5, dash='dot'),
            name='City mean (all 63 sectors)',
        ))
 
        # Benchmark polygon (fixed green reference)
        fig_radar.add_trace(go.Scatterpolar(
            r=b_vals + [b_vals[0]], theta=radar_label + [radar_label[0]],
            fill='toself', fillcolor='rgba(0,255,136,0.08)',
            line=dict(color='#00ff88', width=2, dash='dash'),
            name=f'Benchmark — UTFVI Excellent (n={bench_mask.sum()})',
        ))
 
        # Selected sector polygon (changes every time)
        sec_color = '#ff6b6b' if selected_sector in GROUP_A_SECTORS else '#4da6ff'
        sec_fill  = 'rgba(255,107,107,0.18)' if selected_sector in GROUP_A_SECTORS \
                    else 'rgba(77,166,255,0.15)'
        fig_radar.add_trace(go.Scatterpolar(
            r=s_vals + [s_vals[0]], theta=radar_label + [radar_label[0]],
            fill='toself', fillcolor=sec_fill,
            line=dict(color=sec_color, width=2.5),
            name=f'Sector {selected_sector}',
        ))
 
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor='#334466', color='#aaaaaa',
                                tickvals=[0.25, 0.5, 0.75],
                                ticktext=['25%', '50%', '75%']),
                bgcolor='#0d1b2a',
                angularaxis=dict(linecolor='#334466', gridcolor='#334466'),
            ),
            template='plotly_dark', paper_bgcolor='#0d1b2a',
            margin=dict(l=60, r=60, t=50, b=50), height=360,
            legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='center', x=0.5),
            showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)
 
        # ── Per-variable deviation table (the part that changes per sector) ───
        if len(raw_row):
            raw = raw_row.iloc[0]
            gap_data = {
                'Variable'       : ['LST (°C)', 'NDVI', 'TCI', 'Population'],
                'Sector value'   : [f"{raw['LST']:.2f}", f"{raw['NDVI']:.4f}",
                                    f"{raw['TCI']:.3f}", f"{int(raw['Population']):,}"],
                'Benchmark mean' : [f"{bench_raw['LST']:.2f}", f"{bench_raw['NDVI']:.4f}",
                                    f"{bench_raw['TCI']:.3f}", f"{int(bench_raw['Population']):,}"],
                'Gap'            : [
                    f"{raw['LST'] - bench_raw['LST']:+.2f}°C",
                    f"{raw['NDVI'] - bench_raw['NDVI']:+.4f}",
                    f"{raw['TCI'] - bench_raw['TCI']:+.3f}",
                    f"{int(raw['Population'] - bench_raw['Population']):+,}",
                ],
            }
            gap_df = pd.DataFrame(gap_data)
            st.dataframe(gap_df, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — SHAP IMPROVEMENT GUIDE
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("## SHAP Improvement Guidelines — Tensioned Sectors")
    st.markdown(
        "Tensioned sectors fail **both** benchmarks simultaneously: UTFVI ≥ 0 "
        "(active UHI effect) **and** composite score above city mean. "
        "Guidelines are derived from the Random Forest SHAP analysis. "
        "All sectors would reach **UTFVI Excellent** upon meeting the target LST."
    )

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tensioned sectors", str(len(guide)))
    m2.metric("Benchmark LST", f"{BENCH_LST:.2f}°C")
    m3.metric("Avg LST gap", f"+{guide['LST_gap'].mean():.2f}°C")
    m4.metric("Max LST gap", f"+{guide['LST_gap'].max():.2f}°C")

    st.markdown("---")

    # ── Priority bar chart ────────────────────────────────────────────────────
    st.markdown("### Priority Ranking — Intervention Urgency")
    st.markdown(
        "The **Priority Index** weights the importance of the LST gap and the Multi-variable Composite Score gap into a unified measure.\n"
        
    )

    guide_sorted = guide.sort_values('Priority_rank')

    INTERV_COLORS = {
        'Green infrastructure (NDVI)'              : '#00ff88',
        'Built-up reduction / cool surfaces (NDBI)': '#ffd700',
        'Traffic management (TCI)'                 : '#ff6b6b',
        'Density management'                       : '#a78bfa',
        'Mixed / social (SC%)'                     : '#fb923c',
    }

    # Calculate mean composite score across all sectors
    mean_composite_score = dev['composite_score'].mean()
    
    bar_colors = [INTERV_COLORS.get(t, '#555577') for t in guide_sorted['Intervention_type']]

    fig_prio = go.Figure()
    # compute composite gap (sector composite - mean composite) for hover display
    guide_sorted = guide_sorted.copy()
    guide_sorted['Composite_gap'] = guide_sorted['Composite_current'] - mean_composite_score
    
    # Pre-format composite gap to 3 decimals with sign
    composite_gap_formatted = [f"{g:+.3f}" for g in guide_sorted['Composite_gap']]

    fig_prio.add_trace(go.Bar(
        x=[f"S{s} (#{r})" for s, r in zip(guide_sorted['Sector'], guide_sorted['Priority_rank'])],
        y=guide_sorted['Priority_index'],
        marker_color=bar_colors,
        marker_line_color='#1a1a2e', marker_line_width=0.5,
        text=[f"{g:+.3f}°C" for g in guide_sorted['LST_gap']],
        textposition='outside', textfont=dict(color='white', size=10),
        customdata=np.column_stack((guide_sorted['Intervention_type'], 
                                     guide_sorted['Composite_current'], 
                                     composite_gap_formatted)),
        hovertemplate=(
            "<b>Sector %{x}</b><br>"
            "Priority index: %{y:.3f}<br>"
            "LST gap: %{text}<br>"
            "M-Var Composite gap: %{customdata[2]}<br>"
            "%{customdata[0]}<br>"
            "<extra></extra>"
        ),
    ))

    fig_prio.update_layout(
        template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#0d1b2a',
        margin=dict(l=20,r=20,t=20,b=60), height=380,
        barmode='group',
        xaxis=dict(title='Sector',
                   tickangle=45, gridcolor='#1e2d3d'),
        yaxis=dict(title='Priority index (0–1)', gridcolor='#1e2d3d', range=[0,1.1]),
        legend=dict(orientation='h', yanchor='top', y=-0.25, title='Intervention type'),
        showlegend=False,
    )
    st.plotly_chart(fig_prio, use_container_width=True)

    # ── Detailed guidelines table ─────────────────────────────────────────────
    st.markdown("### Guidelines")

    display_cols = {
        'Sector'           : 'Sector',
        'Priority_rank'    : 'Priority',
        'LST_current'      : 'LST now (°C)',
        'LST_gap'          : 'Gap to bench (°C)',
        'ΔNDVI_required'   : 'ΔNDVI',
        'ΔTCI_required'    : 'ΔTCI',
        'NDVI_target'      : 'NDVI target',
        'TCI_target'       : 'TCI target',
        'UTFVI_class_cur'  : 'Current class',
        'UTFVI_class_tgt'  : 'Target class',
        'Composite_change' : 'Δ Composite',
        'Intervention_type': 'Intervention type',
    }

    tbl = guide_sorted[list(display_cols.keys())].rename(columns=display_cols).reset_index(drop=True)

    # Format numerics
    for col in ['LST now (°C)', 'Gap to bench (°C)', 'ΔNDVI needed',
                'ΔTCI needed', 'NDVI target', 'TCI target', 'Δ Composite']:
        if col in tbl.columns:
            tbl[col] = tbl[col].map(lambda x: f"{x:+.3f}" if col in ['ΔNDVI needed','ΔTCI needed','Δ Composite'] else f"{x:.3f}")

    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ── Scatter: LST gap vs ΔNDVI required ────────────────────────────────────
    st.markdown("### LST Gap vs Required NDVI Increase")
    st.markdown("Each point represents a tensioned sector. Larger LST gap → more vegetation needed.")

    fig_sc = px.scatter(
        guide,
        x='LST_gap',
        y='ΔNDVI_required',
        color='Intervention_type',
        color_discrete_map=INTERV_COLORS,
        text='Sector',
        size='UTFVI_current',
        size_max=25,
        labels={'LST_gap': 'LST gap to benchmark (°C)',
                'ΔNDVI_required': 'Required NDVI increase',
                'Intervention_type': 'Intervention type',
                'UTFVI_current': 'UTFVI'},
        template='plotly_dark',
    )
    fig_sc.update_traces(textposition='top center', textfont_size=10)
    fig_sc.update_layout(
        paper_bgcolor='#0d1117', plot_bgcolor='#0d1b2a',
        margin=dict(l=20,r=20,t=20,b=40), height=350,
        xaxis=dict(gridcolor='#1e2d3d'),
        yaxis=dict(gridcolor='#1e2d3d'),
    )
    st.plotly_chart(fig_sc, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — BENCHMARK COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("## Benchmark Comparison")
    st.markdown(
        "All residential sectors compared against the UTFVI Excellent benchmark "
        "(mean of 30 sectors with UTFVI < 0). Benchmark values: "
        f"LST = **{bench.get('LST',0):.2f}°C**, "
        f"NDVI = **{bench.get('NDVI',0):.4f}**, "
        f"TCI = **{bench.get('TCI',0):.4f}**. "
        "The Multi-Variable Composite Score benchmark is the mean of the UTFVI < 0.0 composite scores, following the same criteria as with the LST benchmark."
    )

    # ── Composite score bar chart coloured by UTFVI class ────────────────────
    st.markdown("### Composite Vulnerability Score")

    dev_sorted = dev.sort_values('composite_score').reset_index(drop=True)

    UTFVI_COLORS_HEX = {
        'Excellent': '#2c7bb6',
        'Good'     : '#74add1',
        'Normal'   : '#ffffbf',
        'Bad'      : '#fdae61',
        'Worse'    : '#f46d43',
        'Worst'    : '#d73027',
        'N/A'      : '#555577',
    }

    bar_c = [UTFVI_COLORS_HEX.get(cls, '#555577')
             for cls in dev_sorted.get('UTFVI_class', ['N/A']*len(dev_sorted))]

    bench_threshold = dev['composite_score'].mean()  # Benchmark threshold for composite score

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=dev_sorted['Sector_nam'],
        y=dev_sorted['composite_score'],
        marker_color=bar_c,
        marker_line_color='#1a1a2e', marker_line_width=0.3,
        showlegend=False,
        hovertemplate=(
            "<b>Sector %{x}</b><br>"
            "Composite score: %{y:.3f}<br>"
            "<extra></extra>"
        ),
    ))
    fig_bar.add_hline(y=bench_threshold, line_color="#eceef0", line_dash='dash', line_width=1.5,
                      annotation_text=f"Composite Score Benchmark ({bench_threshold:.3f})",
                      annotation_font_color="#eeeff0")

    # Colour legend with UTFVI ranges
    legend_entries = [
        ('Excellent', '#2c7bb6', 'UTFVI ≤ 0'),
        ('Good', '#74add1', '0 < UTFVI ≤ 0.005'),
        ('Normal', '#ffffbf', '0.005 < UTFVI ≤ 0.010'),
        ('Bad', '#fdae61', '0.010 < UTFVI ≤ 0.015'),
        ('Worse', '#f46d43', '0.015 < UTFVI ≤ 0.020'),
        ('Worst', '#d73027', 'UTFVI > 0.020'),
    ]
    
    for cls, clr, rng in legend_entries:
        if cls in dev_sorted.get('UTFVI_class', pd.Series(dtype=str)).values:
            fig_bar.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                name=f'{cls} — {rng}',
                marker=dict(size=10, color=clr),
                showlegend=True,
                hoverinfo='skip'
            ))

    fig_bar.update_layout(
        template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#0d1b2a',
        margin=dict(l=20,r=20,t=20,b=100), height=450,
        xaxis=dict(title='Sector', tickangle=45, gridcolor='#1e2d3d'),
        yaxis=dict(title='Composite vulnerability score', gridcolor='#1e2d3d'),
        legend=dict(
            orientation='h', 
            yanchor='bottom', 
            y=-0.28,
            xanchor='center',
            x=0.1,
            title='UTFVI class',
            title_font=dict(size=12),
        ),
        showlegend=True,
        hovermode='x unified',
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Deviation table ───────────────────────────────────────────────────────
    st.markdown("### LST Deviation from Benchmark")

    # Place controls above the table for better layout and state management
    show_only_above = st.checkbox("Show only sectors above benchmark", value=False, key="tab4_filter")
    sort_by = st.selectbox("Sort by:", ["Sector name", "LST deviation", "Composite score"], key="tab4_sort")

    # Create a display copy and apply filters/sorts
    dev_display = dev.copy()
    
    # Apply filter
    if show_only_above:
        if 'LST_dev' in dev_display.columns:
            dev_display = dev_display[dev_display['LST_dev'] > 0]
    
    # Apply sort
    if sort_by == "Sector name":
        # Custom numeric sort: numeric sectors first (sorted numerically), then non-numeric (alphabetically)
        def sector_sort_key(sector):
            try:
                return (0, int(sector))  # Numeric sectors: (0, numeric_value)
            except ValueError:
                return (1, sector)  # Non-numeric sectors: (1, string_value)
        
        dev_display = dev_display.copy()
        dev_display['sort_key'] = dev_display['Sector_nam'].apply(sector_sort_key)
        dev_display = dev_display.sort_values('sort_key', ascending=True).drop('sort_key', axis=1)
    elif sort_by == "LST deviation":
        if 'LST_dev' in dev_display.columns:
            dev_display = dev_display.sort_values('LST_dev', ascending=False)
        else:
            dev_display = dev_display.sort_values('Sector_nam')
    elif sort_by == "Composite score":
        dev_display = dev_display.sort_values('composite_score', ascending=False)

    # Prepare columns to display
    show_cols = ['Sector_nam','UTFVI_class','LST','NDVI','TCI','composite_score']
    if 'LST_dev' in dev_display.columns:
        show_cols.insert(3,'LST_dev')

    st.dataframe(
        dev_display[show_cols].rename(columns={
            'Sector_nam'     : 'Sector',
            'UTFVI_class'    : 'UTFVI class',
            'LST'            : 'LST (°C)',
            'LST_dev'        : 'LST dev (°C)',
            'composite_score': 'Composite score',
        }).reset_index(drop=True),
        use_container_width=True, hide_index=True, height=400,
    )

    # ── Scatter: LST vs NDVI with UTFVI colour ────────────────────────────────
    st.markdown("### LST vs NDVI")

    scatter_df = params.copy()  # Include all sectors in scatter
    scatter_df['is_benchmark'] = scatter_df['UTFVI'] < 0
    scatter_df['label'] = scatter_df['Sector_nam'].apply(
        lambda s: f"S{s}**" if s in GROUP_A_SECTORS else f"S{s}"
    )

    fig_lst_ndvi = px.scatter(
        scatter_df,
        x='NDVI', y='LST',
        color='UTFVI_class',
        color_discrete_map=UTFVI_COLORS_HEX,
        text='label',
        size_max=12,
        labels={'NDVI': 'NDVI (vegetation index)', 'LST': 'LST (°C)', 'UTFVI_class': 'UTFVI class'},
        template='plotly_dark',
        hover_data={'UTFVI_class': True, 'TCI': True},
    )
    fig_lst_ndvi.update_traces(textposition='top center', textfont_size=7, marker_size=9)

    # Benchmark target point
    fig_lst_ndvi.add_trace(go.Scatter(
        x=[bench.get('NDVI', 0.46)], y=[BENCH_LST],
        mode='markers',
        marker=dict(symbol='star', size=18, color='#00ff88', line_width=1),
        name='Benchmark target',
        hovertemplate=f"Benchmark<br>NDVI={bench.get('NDVI',0):.4f}<br>LST={BENCH_LST:.2f}°C<extra></extra>",
    ))

    fig_lst_ndvi.update_layout(
        paper_bgcolor='#0d1117', plot_bgcolor='#0d1b2a',
        margin=dict(l=20,r=20,t=20,b=40), height=400,
        xaxis=dict(gridcolor='#1e2d3d'),
        yaxis=dict(gridcolor='#1e2d3d'),
    )
    st.plotly_chart(fig_lst_ndvi, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555577;font-size:12px'>"
    "Chandigarh Multi-Layer Map  |  Elias Ruiz Sabater  |  MMDU  |  Supervisor: Dr. Rajiv Chechi  | February - April 2026<br>"
    "Data: Google Earth Engine (Landsat 8/9), Chandigarh Administration, HDX, OpenStreetMap"
    "</div>",
    unsafe_allow_html=True,
)
