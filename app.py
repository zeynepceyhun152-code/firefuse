import streamlit as st, pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go, plotly.io as pio
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from shapely.geometry import box
from shapely import contains_xy
from scipy.stats import theilslopes, kendalltau

st.set_page_config(page_title="FireFuse | Harmonized Burning Activity Calendar", page_icon="🔥", layout="wide")
pio.templates.default = "plotly_white"
pio.templates["plotly_white"].layout.font.family = "-apple-system, BlinkMacSystemFont, Helvetica Neue, sans-serif"
pio.templates["plotly_white"].layout.font.color = "#1d1d1f"
st.markdown("""<style>
html, body, .stApp, [class*="css"], button, input {font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text","Helvetica Neue",Inter,sans-serif!important;}
.stApp{background:#f5f5f7}
section[data-testid="stSidebar"]{background:rgba(255,255,255,.75);backdrop-filter:saturate(180%) blur(20px);border-right:1px solid rgba(0,0,0,.06)}
.block-container{padding-top:2.2rem;max-width:1400px}
h1{font-size:2.5rem!important;font-weight:700!important;letter-spacing:-.03em;color:#1d1d1f!important;margin-bottom:.2rem!important}
h5{font-weight:600!important;color:#1d1d1f!important;letter-spacing:-.01em}
[data-testid="stCaptionContainer"] p{color:#6e6e73!important;font-size:.95rem}
[data-testid="stMetric"]{background:#fff;border-radius:18px;padding:16px 18px;box-shadow:0 2px 14px rgba(0,0,0,.06);border:none}
[data-testid="stMetricLabel"] p{font-size:.78rem;color:#6e6e73!important;font-weight:500;white-space:normal!important}
[data-testid="stMetricLabel"] div{overflow:visible!important}
[data-testid="stMetricValue"]{font-size:1.65rem;font-weight:600;color:#1d1d1f;letter-spacing:-.02em}
[data-testid="stMetricValue"] div{overflow:visible!important;text-overflow:clip!important}
iframe{border-radius:18px;box-shadow:0 2px 14px rgba(0,0,0,.08)}
[data-testid="stPlotlyChart"]{background:#fff;border-radius:18px;padding:10px;box-shadow:0 2px 14px rgba(0,0,0,.06)}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:rgba(0,0,0,.05);padding:4px;border-radius:12px;width:fit-content}
.stTabs [data-baseweb="tab"]{border-radius:9px;padding:6px 18px;height:auto;font-weight:500}
.stTabs [aria-selected="true"]{background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.12)}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}
.stDownloadButton button{border-radius:980px;background:#0071e3;color:#fff!important;border:none;font-weight:500}
.stDownloadButton button:hover{background:#0077ed}
</style>""", unsafe_allow_html=True)

BLUE, INDIGO, ORANGE, RED, GRAY = "#0071e3", "#5e5ce6", "#ff9f0a", "#ff3b30", "#86868b"

TXT = {
 "en": dict(tag="One consistent MODIS + VIIRS hotspot record, 2001 → present", aoi="Area of interest", preset="Preset",
    click="Click on map", latlon="Lat/Lon box", region="Region", size="Area size (km)", years="Years", logc="Log color scale",
    data="Data: NASA FIRMS MODIS C6.1 (Terra/Aqua) + VIIRS S-NPP 375 m · ESA CCI Land Cover. Built for NASA Space Apps 2026.",
    title="Harmonized Burning Activity Calendar",
    caption="Pick an area. See 25 years of fire activity, anomalies against history, and download the data.",
    click_info="Click anywhere on the map to analyze that area.",
    nodata="No fire activity recorded here. Try a larger area or another spot.",
    k_hot="Hotspots", k_peak="Peak month", k_top="Most active year", vs="vs median", k_missed="Fires MODIS missed", saw="MODIS saw only",
    per_year="Hotspots per year", legend="Indigo: harmonized from MODIS (whiskers = 80% range) · Orange: observed VIIRS",
    cells="grid cells (0.25°) · VIIRS-equivalent counts",
    tabs=["Calendar", "Anomalies", "Season trends", "Harmonization", "Download"],
    cal_cap="Above the blue line: harmonized from MODIS · Below: observed VIIRS. Hover any cell for its 80% uncertainty range. Colorblind-safe ColorBrewer YlOrRd ramp.",
    cmp="Compare year", med="Historical median", iqr="Historical 25–75%", hs="Hotspots", median="Median", anom="Anomaly %",
    anom_cap="Red bars are above the historical 75th percentile (unusual activity).",
    trend_t="Is the fire season changing? Trend per calendar month",
    trend_cap="Theil–Sen slope as % of the monthly median, per decade. Solid bars are statistically significant (Mann–Kendall p < 0.05); faded bars are not.",
    trend_none="No statistically significant monthly trend in this area.", up="Significant increase in", down="Significant decrease in",
    raw_m="MODIS raw", raw_v="VIIRS raw", harm="Harmonized", band="80% uncertainty range",
    seen="Detected by MODIS", hidden="Missed by MODIS (estimated)", hid_t="What MODIS could not see",
    method="""**Method.** Both sensors are gridded to 0.25° × monthly cells. A LightGBM Poisson model learns the MODIS → VIIRS
mapping from the stable-orbit overlap years 2012–2021 (features: Terra/Aqua counts, fire radiative power, neighbourhood activity,
seasonality, location, MODIS viewing angle, ESA CCI land cover). A separate Terra-only model covers 2001–Jun 2002, before Aqua.
Uncertainty comes from a 10-member ensemble trained on bootstrap-resampled years.
**Held-out test (2020–2025):** national error raw MODIS 71% → simple scaling 14% → FireFuse 10.0% (5.6% in stable-orbit years);
regional (2°) R² 0.72 → 0.86 → 0.93.""",
    dl_cal="Calendar (CSV)", dl_ser="Monthly series (CSV)", dl_raw="Raw hotspots (CSV)", raw_n="raw hotspots in this area",
    months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']),
 "es": dict(tag="Un registro consistente de focos de calor MODIS + VIIRS, 2001 → hoy", aoi="Área de interés", preset="Predefinida",
    click="Clic en el mapa", latlon="Caja lat/lon", region="Región", size="Tamaño del área (km)", years="Años", logc="Escala de color logarítmica",
    data="Datos: NASA FIRMS MODIS C6.1 (Terra/Aqua) + VIIRS S-NPP 375 m · ESA CCI Land Cover. Hecho para NASA Space Apps 2026.",
    title="Calendario Armonizado de Actividad de Quema",
    caption="Elija un área. Vea 25 años de actividad de fuego, anomalías frente a la historia y descargue los datos.",
    click_info="Haga clic en cualquier lugar del mapa para analizar esa área.",
    nodata="No hay actividad de fuego registrada aquí. Pruebe un área más grande u otro lugar.",
    k_hot="Focos de calor", k_peak="Mes pico", k_top="Año más activo", vs="vs mediana", k_missed="Fuegos no vistos por MODIS", saw="MODIS vio solo",
    per_year="Focos de calor por año", legend="Índigo: armonizado desde MODIS (bigotes = rango 80%) · Naranja: VIIRS observado",
    cells="celdas (0,25°) · conteos equivalentes a VIIRS",
    tabs=["Calendario", "Anomalías", "Tendencias", "Armonización", "Descargas"],
    cal_cap="Sobre la línea azul: armonizado desde MODIS · Debajo: VIIRS observado. Pase el cursor sobre una celda para ver su rango de incertidumbre del 80%. Paleta YlOrRd apta para daltonismo.",
    cmp="Comparar año", med="Mediana histórica", iqr="Histórico 25–75%", hs="Focos", median="Mediana", anom="Anomalía %",
    anom_cap="Las barras rojas están por encima del percentil 75 histórico (actividad inusual).",
    trend_t="¿Está cambiando la temporada de fuego? Tendencia por mes",
    trend_cap="Pendiente de Theil–Sen como % de la mediana mensual, por década. Barras sólidas = significativas (Mann–Kendall p < 0,05); tenues = no significativas.",
    trend_none="No hay tendencias mensuales significativas en esta área.", up="Aumento significativo en", down="Disminución significativa en",
    raw_m="MODIS crudo", raw_v="VIIRS crudo", harm="Armonizado", band="Rango de incertidumbre 80%",
    seen="Detectado por MODIS", hidden="No detectado por MODIS (estimado)", hid_t="Lo que MODIS no pudo ver",
    method="""**Método.** Ambos sensores se agregan en celdas de 0,25° × mes. Un modelo LightGBM Poisson aprende la relación MODIS → VIIRS
en los años de órbita estable 2012–2021 (variables: conteos Terra/Aqua, potencia radiativa, actividad vecina, estacionalidad,
ubicación, ángulo de observación MODIS, cobertura del suelo ESA CCI). Un modelo solo-Terra cubre 2001–jun 2002, antes de Aqua.
La incertidumbre proviene de un conjunto de 10 modelos entrenados con años remuestreados (bootstrap).
**Prueba independiente (2020–2025):** error nacional MODIS crudo 71% → escalado simple 14% → FireFuse 10,0% (5,6% en años de órbita estable);
R² regional (2°) 0,72 → 0,86 → 0,93.""",
    dl_cal="Calendario (CSV)", dl_ser="Serie mensual (CSV)", dl_raw="Focos crudos (CSV)", raw_n="focos crudos en esta área",
    months=['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']),
}

RES = 0.25
ECOLS = [f"e{b}" for b in range(10)]
PRESETS = {  # (south, west, north, east)
    "Chilecito, La Rioja": (-29.7, -68.0, -28.6, -67.0),
    "Paraná Delta": (-34.3, -60.8, -32.6, -58.4),
    "Corrientes (Iberá)": (-29.5, -58.3, -27.4, -56.0),
    "All Argentina": (-55.1, -73.6, -21.7, -53.6),
}
TRANSP = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

def click_box(lat, lon, km):
    dlat = km / 2 / 111.0
    dlon = dlat / max(np.cos(np.radians(lat)), 0.2)
    return box(lon - dlon, lat - dlat, lon + dlon, lat + dlat)

def interval(point, E):
    with np.errstate(all="ignore"):
        mean = E.mean(axis=1)
        lo = np.where(mean > 0, np.percentile(E, 10, axis=1) / mean, 1.0)
        hi = np.where(mean > 0, np.percentile(E, 90, axis=1) / mean, 1.0)
    return point * lo, point * hi

@st.cache_resource
def load():
    h = pd.read_parquet("harmonized.parquet")
    for c in ECOLS:
        if c not in h: h[c] = h.harm_n
    p = pd.read_parquet("points.parquet")
    p["acq_date"] = pd.to_datetime(p["acq_date"])
    cells = h[["gy","gx","lat","lon"]].drop_duplicates().reset_index(drop=True)
    heat = h.groupby(["lat","lon"]).harm_n.sum().reset_index()
    heat["w"] = np.log1p(heat.harm_n) / np.log1p(heat.harm_n).max()
    return h, p, cells, heat[["lat","lon","w"]].values.tolist()
H, P, CELLS, HEAT = load()
YMAX = int(H.ym.max() // 100)

# ---------- sidebar ----------
lang = st.sidebar.radio("Language", ["English", "Español"], horizontal=True, label_visibility="collapsed")
T = TXT["en" if lang == "English" else "es"]
MONTHS = T["months"]
st.sidebar.title("FireFuse")
st.sidebar.caption(T["tag"])
mode = st.sidebar.radio(T["aoi"], ["preset", "click", "latlon"], format_func=lambda x: T[x])
geom = None
if mode == "preset":
    s, w, n, e = PRESETS[st.sidebar.selectbox(T["region"], list(PRESETS))]
    geom = box(w, s, e, n)
elif mode == "latlon":
    c1, c2 = st.sidebar.columns(2)
    s, n = c1.number_input("S", value=-29.7), c2.number_input("N", value=-28.6)
    w, e = c1.number_input("W", value=-68.0), c2.number_input("E", value=-67.0)
    geom = box(w, s, e, n)
else:
    km = st.sidebar.slider(T["size"], 25, 400, 100, step=25)
    if "clk" in st.session_state:
        geom = click_box(*st.session_state.clk, km)
y0, y1 = st.sidebar.slider(T["years"], 2001, YMAX, (2001, YMAX))
log_color = st.sidebar.toggle(T["logc"], value=True)
st.sidebar.markdown("---")
st.sidebar.caption(T["data"])

# ---------- header ----------
st.title(T["title"])
st.caption(T["caption"])
kpi_box = st.container()
left, right = st.columns([3, 2], gap="large")

# ---------- map ----------
if geom is not None:
    bb = geom.bounds
    center = [(bb[1] + bb[3]) / 2, (bb[0] + bb[2]) / 2]
    zoom = 4 if (bb[2] - bb[0]) > 10 else (6 if mode == "click" else 7)
else:
    center, zoom = [-38, -64], 4
fmap = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                 attr="Esri World Imagery", name="Satellite", show=True).add_to(fmap)
folium.TileLayer("OpenStreetMap", name="Streets", show=False).add_to(fmap)
HeatMap(HEAT, radius=12, blur=15, min_opacity=0.15, max_zoom=12, name="Fire density 2001–present",
        gradient={0.3: "#ffffb2", 0.55: "#fecc5c", 0.8: "#fd8d3c", 1: "#e31a1c"}).add_to(fmap)
if geom is not None:
    folium.Rectangle([[bb[1], bb[0]], [bb[3], bb[2]]], color="#0a84ff", weight=3, fill=False).add_to(fmap)
folium.LayerControl(collapsed=True).add_to(fmap)
with left:
    out = st_folium(fmap, height=430, use_container_width=True, key="map", returned_objects=["last_clicked"])
clk = (out or {}).get("last_clicked")
if mode == "click" and clk:
    new = (round(clk["lat"], 4), round(clk["lng"], 4))
    if st.session_state.get("clk") != new:
        st.session_state.clk = new
        st.rerun()
if geom is None:
    with right: st.info(T["click_info"])
    st.stop()

# ---------- select + aggregate ----------
inside = contains_xy(geom, CELLS.lon.values, CELLS.lat.values)
if not inside.any():
    bb2, h2 = geom.bounds, RES / 2
    inside = ((CELLS.lon + h2 >= bb2[0]) & (CELLS.lon - h2 <= bb2[2]) & (CELLS.lat + h2 >= bb2[1]) & (CELLS.lat - h2 <= bb2[3])).values
sel_cells = CELLS[inside][["gy","gx"]]
if sel_cells.empty:
    with right: st.warning(T["nodata"])
    st.stop()
sel = H.merge(sel_cells, on=["gy","gx"])
sel = sel[sel.ym.between(y0*100 + 1, y1*100 + 12)]

all_ym = [y*100 + m for y in range(y0, y1 + 1) for m in range(1, 13)]
cols = ["m_n","v_n","harm_n"] + ECOLS
ts = sel.groupby("ym")[cols].sum(min_count=1).reindex(all_ym)
ts[["m_n","harm_n"] + ECOLS] = ts[["m_n","harm_n"] + ECOLS].fillna(0)
ts["year"], ts["month"] = ts.index // 100, ts.index % 100
ts.loc[ts.index > H.ym.max(), ["harm_n"] + ECOLS] = np.nan
ts.loc[ts.index >= 202601, "m_n"] = np.nan
ts["lo"], ts["hi"] = interval(ts.harm_n.values, ts[ECOLS].values)

cal = ts.pivot_table(index="year", columns="month", values="harm_n", aggfunc="sum", dropna=False).reindex(columns=range(1, 13)).round(0)
cal_lo = ts.pivot_table(index="year", columns="month", values="lo", aggfunc="sum", dropna=False).reindex(index=cal.index, columns=range(1, 13))
cal_hi = ts.pivot_table(index="year", columns="month", values="hi", aggfunc="sum", dropna=False).reindex(index=cal.index, columns=range(1, 13))

yt = ts.groupby("year")[["m_n","v_n","harm_n"] + ECOLS].sum(min_count=1)
yt["lo"], yt["hi"] = interval(yt.harm_n.values, yt[ECOLS].values)

both = ts.m_n.notna() & ts.harm_n.notna()
h_sum, m_sum = ts.loc[both, "harm_n"].sum(), ts.loc[both, "m_n"].sum()
missed = max(h_sum - m_sum, 0)

# ---------- KPIs ----------
med_by_month = cal.median()
last = ts[ts.harm_n.notna()].iloc[-1]
lm_med = cal.drop(index=int(last.year), errors="ignore")[int(last.month)].median()
with kpi_box:
    cA, cB, cC, cD, cE = st.columns(5)
    cA.metric(T["k_hot"], f"{int(np.nansum(cal.values)):,}")
    cB.metric(T["k_peak"], MONTHS[int(med_by_month.idxmax()) - 1])
    cC.metric(T["k_top"], int(cal.sum(axis=1).idxmax()))
    cD.metric(f"{MONTHS[int(last.month)-1]} {int(last.year)}", f"{int(last.harm_n):,}",
              f"{(last.harm_n - lm_med) / max(lm_med, 1) * 100:+.0f}% {T['vs']}", delta_color="inverse")
    cE.metric(T["k_missed"], f"{int(missed):,}", f"{T['saw']} {m_sum / max(h_sum, 1) * 100:.0f}%", delta_color="off")

with right:
    st.markdown("##### " + T["per_year"])
    pre = yt.index < 2012
    err_hi = np.where(pre, yt.hi - yt.harm_n, 0); err_lo = np.where(pre, yt.harm_n - yt.lo, 0)
    fy = go.Figure(go.Bar(x=yt.index, y=yt.harm_n, marker_color=np.where(pre, INDIGO, ORANGE),
                          error_y=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo, color=GRAY, thickness=1.2, width=2),
                          hovertemplate="%{x}: %{y:,.0f}<extra></extra>"))
    fy.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0), showlegend=False, **TRANSP)
    st.plotly_chart(fy, use_container_width=True)
    st.caption(f"{T['legend']} · {len(sel_cells)} {T['cells']}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(T["tabs"])

with tab1:
    z = np.log1p(cal) if log_color else cal
    fig = px.imshow(z, x=MONTHS, y=cal.index, color_continuous_scale="YlOrRd", aspect="auto", labels=dict(x="", y="", color=""))
    cd = np.dstack([cal_lo.values, cal_hi.values]).tolist()
    fig.update_traces(text=cal.fillna(0).astype(int).values, texttemplate="%{text}", textfont_size=11, customdata=cd,
                      hovertemplate="%{y} %{x}: %{text}<br>80%: %{customdata[0]:.0f} – %{customdata[1]:.0f}<extra></extra>",
                      xgap=2, ygap=2)
    if y0 < 2012 <= y1:
        fig.add_hline(y=2011.5, line_color=BLUE, line_width=3)
    fig.update_layout(height=max(450, 26 * len(cal)), coloraxis_showscale=False,
                      yaxis=dict(autorange="reversed", dtick=1), xaxis=dict(side="top"), margin=dict(t=30, l=10, r=10, b=10), **TRANSP)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(T["cal_cap"])

with tab2:
    yr = st.selectbox(T["cmp"], sorted(cal.index, reverse=True))
    others = cal.drop(index=yr, errors="ignore")
    med, q25, q75 = others.median(), others.quantile(.25), others.quantile(.75)
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=MONTHS, y=q75, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    f2.add_trace(go.Scatter(x=MONTHS, y=q25, fill="tonexty", fillcolor="rgba(0,0,0,.06)", line=dict(width=0), name=T["iqr"]))
    f2.add_trace(go.Scatter(x=MONTHS, y=med, line=dict(color=GRAY, dash="dash"), name=T["med"]))
    f2.add_trace(go.Bar(x=MONTHS, y=cal.loc[yr], marker_color=np.where(cal.loc[yr] > q75, RED, "#ffcc80"), name=str(yr)))
    f2.update_layout(height=420, yaxis_title=T["hs"], legend=dict(orientation="h", y=1.1), **TRANSP)
    st.plotly_chart(f2, use_container_width=True)
    anom = ((cal.loc[yr] - med) / med.clip(lower=1) * 100).round(0)
    st.dataframe(pd.DataFrame({T["hs"]: cal.loc[yr].values, T["median"]: med.values, T["anom"]: anom.values}, index=MONTHS).T,
                 use_container_width=True)
    st.caption(T["anom_cap"])

with tab3:
    st.markdown("##### " + T["trend_t"])
    rows = []
    for mth in range(1, 13):
        s = cal[mth].dropna()
        if len(s) >= 8 and s.sum() > 0:
            slope = theilslopes(s.values, s.index.values)[0]
            p = kendalltau(s.index.values, s.values).pvalue
            rows.append((mth, slope * 10 / max(s.median(), 1) * 100, p if np.isfinite(p) else 1.0))
        else:
            rows.append((mth, 0.0, 1.0))
    tr = pd.DataFrame(rows, columns=["m", "pct", "p"])
    sig = (tr.p < 0.05).values
    f4 = go.Figure(go.Bar(x=MONTHS, y=tr.pct, marker_color=np.where(tr.pct > 0, RED, BLUE),
                          marker_opacity=np.where(sig, 1.0, 0.3), customdata=tr.p,
                          hovertemplate="%{x}: %{y:+.0f}% / decade<br>p = %{customdata:.3f}<extra></extra>"))
    f4.add_hline(y=0, line_color="#c7c7cc")
    f4.update_layout(height=380, yaxis_title="% / decade", **TRANSP)
    st.plotly_chart(f4, use_container_width=True)
    ups = [MONTHS[m - 1] for m, pc, pv in rows if pv < 0.05 and pc > 0]
    downs = [MONTHS[m - 1] for m, pc, pv in rows if pv < 0.05 and pc < 0]
    if ups: st.error(f"{T['up']}: {', '.join(ups)}")
    if downs: st.success(f"{T['down']}: {', '.join(downs)}")
    if not ups and not downs: st.info(T["trend_none"])
    st.caption(T["trend_cap"])

with tab4:
    f3 = go.Figure()
    pre_y = yt[yt.index < 2012]
    f3.add_trace(go.Scatter(x=pre_y.index, y=pre_y.hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    f3.add_trace(go.Scatter(x=pre_y.index, y=pre_y.lo, fill="tonexty", fillcolor="rgba(94,92,230,.15)", line=dict(width=0), name=T["band"]))
    f3.add_trace(go.Scatter(x=yt.index, y=yt.m_n, name=T["raw_m"], line=dict(color="#aeaeb2")))
    f3.add_trace(go.Scatter(x=yt.index, y=yt.v_n, name=T["raw_v"], line=dict(color=ORANGE)))
    f3.add_trace(go.Scatter(x=yt.index, y=yt.harm_n, name=T["harm"], line=dict(color=INDIGO, width=3, dash="dash")))
    f3.update_layout(height=400, yaxis_title=T["hs"], legend=dict(orientation="h", y=1.12), **TRANSP)
    st.plotly_chart(f3, use_container_width=True)

    st.markdown("##### " + T["hid_t"])
    yb = yt[yt.index < 2026]
    hid = (yb.harm_n - yb.m_n.fillna(0)).clip(lower=0)
    f5 = go.Figure()
    f5.add_trace(go.Bar(x=yb.index, y=yb.m_n, name=T["seen"], marker_color="#aeaeb2"))
    f5.add_trace(go.Bar(x=yb.index, y=hid, name=T["hidden"], marker_color=RED))
    f5.update_layout(barmode="stack", height=340, legend=dict(orientation="h", y=1.12), **TRANSP)
    st.plotly_chart(f5, use_container_width=True)
    st.markdown(T["method"])

with tab5:
    c1, c2 = st.columns(2)
    c1.download_button(T["dl_cal"], cal.to_csv().encode(), "burning_calendar.csv", "text/csv", use_container_width=True)
    c2.download_button(T["dl_ser"], ts.drop(columns=ECOLS).to_csv().encode(), "monthly_series.csv", "text/csv", use_container_width=True)
    bb3 = geom.bounds
    pts = P[P.longitude.between(bb3[0], bb3[2]) & P.latitude.between(bb3[1], bb3[3]) & P.acq_date.dt.year.between(y0, y1)]
    pts = pts[contains_xy(geom, pts.longitude.values, pts.latitude.values)]
    st.write(f"**{len(pts):,}** {T['raw_n']}")
    st.download_button(T["dl_raw"], pts.to_csv(index=False).encode(), "hotspots.csv", "text/csv", use_container_width=True)
