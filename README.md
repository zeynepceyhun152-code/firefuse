# FireFuse 
**One consistent NASA fire record, 2001 → today.** NASA Space Apps 2026 — Harmonization of MODIS and VIIRS Hot Spots (Kansas City).

MODIS (2000–, retiring) and VIIRS (2012–, ~3× more sensitive) count fires differently, so their records can't be compared. FireFuse learns the MODIS→VIIRS mapping with a LightGBM Poisson model on stable-orbit overlap years (2012–2021), using fire radiative power, neighbourhood activity, seasonality, location, MODIS viewing angle and ESA CCI land cover — then reconstructs 2001–2011 in VIIRS-equivalent terms with a 10-member bootstrap uncertainty ensemble.

**Held-out test (2020–2025):** national error raw MODIS 71% → simple scaling 14% → FireFuse 10.0% (5.6% in stable-orbit years); regional (2°) R² 0.72 → 0.93.

Web app: burning-activity calendar, anomaly detection, season trends, fires MODIS missed, CSV export — English & Español.

Data: NASA FIRMS (MODIS C6.1, VIIRS S-NPP 375 m), ESA CCI Land Cover via Microsoft Planetary Computer.

Team FireFuse — Zeynep Ceyhun, solo. Built with AI assistance (Claude) for frontend design

Run: `pip install -r requirements.txt && streamlit run app.py`
