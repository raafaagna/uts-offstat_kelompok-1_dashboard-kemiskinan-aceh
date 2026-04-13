"""
app.py — Dashboard Analisis & Segmentasi Risiko Kemiskinan Aceh
Jalankan: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

# ── Konfigurasi halaman (WAJIB baris pertama setelah import) ──────────────────
st.set_page_config(
    page_title="Dashboard Kemiskinan Aceh",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Import modul lokal ────────────────────────────────────────────────────────
from data_loader import (
    load_panel, load_ref, load_klaster, load_geojson,
    build_master, hitung_korelasi, kpi_provinsi,
    WARNA_KLASTER, LABEL_KLASTER_ORDER, NAVY, NAVY_PALE, BG_COLOR, TEXT_SEC,
)
from charts import (
    fig_tren_kemiskinan, fig_top5_termiskin, fig_tren_vs_variabel,
    fig_scatter_korelasi, fig_bar_korelasi, fig_peta_klaster,
    fig_profil_klaster, fig_pergerakan_klaster,
)

# ── CSS kustom ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* Sembunyikan elemen bawaan Streamlit */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 0 !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

/* ── Header ── */
.dash-header {
    background: #231aa1;
    color: white;
    padding: 18px 32px 14px;
    margin: 0 -2rem 0 -2rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
}
.dash-eyebrow {
    font-family: 'Sora', sans-serif;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    opacity: 0.65;
    margin-bottom: 4px;
}
.dash-title {
    font-family: 'Sora', sans-serif;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.3px;
    line-height: 1.2;
    margin: 0;
}
.dash-subtitle {
    font-size: 12px;
    opacity: 0.65;
    margin-top: 3px;
}
.header-badges { display: flex; gap: 8px; flex-wrap: wrap; align-items: flex-end; }
.hbadge {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    color: white;
    font-size: 10px;
    padding: 4px 10px;
    border-radius: 20px;
    font-family: 'Sora', sans-serif;
    font-weight: 500;
    white-space: nowrap;
}

/* ── Filter bar ── */
.filter-bar {
    background: white;
    border-bottom: 1px solid #e2e2f0;
    padding: 10px 0 6px 0;
    margin-bottom: 8px;
}
.filter-label {
    font-family: 'Sora', sans-serif;
    font-size: 10px;
    font-weight: 700;
    color: #231aa1;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
}
.active-tags { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px; }
.atag {
    background: #eceafc;
    color: #231aa1;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 500;
    font-family: 'Sora', sans-serif;
    display: inline-block;
}

/* ── Section label ── */
.section-label {
    font-family: 'Sora', sans-serif;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #6b63d6;
    margin: 18px 0 10px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e2e2f0;
    display: inline-block;
}

/* ── KPI cards ── */
.kpi-grid { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 0; align-items: stretch; }
.kpi-card {
    background: white;
    border: 1px solid #e2e2f0;
    border-radius: 12px;
    padding: 14px 18px 12px;
    flex: 1;
    min-width: 140px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(35,26,161,0.07);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 125px;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}
.kpi-navy::before  { background: #231aa1; }
.kpi-green::before { background: #2e7d32; }
.kpi-yellow::before{ background: #f9a825; }
.kpi-muted::before { background: #6b63d6; }
.kpi-red::before   { background: #e53935; }
.kpi-label {
    font-family: 'Sora', sans-serif;
    font-size: 10px;
    color: #9393b0;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 500;
    margin-bottom: 4px;
}
.kpi-value {
    font-family: 'Sora', sans-serif;
    font-size: 26px;
    font-weight: 700;
    color: #1a1a2e;
    line-height: 1.1;
    margin: 2px 0;
}
.kpi-unit { font-size: 13px; font-weight: 400; color: #5a5a7a; }
.kpi-delta-up   { background: #e8f5e9; color: #2e7d32; font-size: 11px; font-weight: 500; padding: 2px 8px; border-radius: 10px; display: inline-block; }
.kpi-delta-down { background: #ffebee; color: #e53935; font-size: 11px; font-weight: 500; padding: 2px 8px; border-radius: 10px; display: inline-block; }
.kpi-delta-neu  { background: #eceafc; color: #231aa1; font-size: 11px; font-weight: 500; padding: 2px 8px; border-radius: 10px; display: inline-block; }

/* ── Insight box ── */
.insight-box {
    background: #eceafc;
    border-left: 3px solid #231aa1;
    border-radius: 0 8px 8px 0;
    padding: 12px 14px;
    font-size: 12.5px;
    color: #1a1a2e;
    line-height: 1.7;
    margin-top: 10px;
}
.insight-title {
    font-family: 'Sora', sans-serif;
    font-size: 10px;
    font-weight: 700;
    color: #231aa1;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.insight-dot {
    width: 14px; height: 14px;
    background: #231aa1;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 9px;
    color: white;
    font-weight: 700;
    flex-shrink: 0;
}
.hl { font-weight: 600; color: #231aa1; }

/* ── Cluster cards ── */
.cluster-card {
    border-radius: 12px;
    padding: 16px;
    border: 1px solid;
    height: 100%;
}
.cluster-card-rentan   { background: #ffebee; border-color: #ef9a9a; }
.cluster-card-transisi { background: #fffde7; border-color: #ffe082; }
.cluster-card-mandiri  { background: #e8f5e9; border-color: #a5d6a7; }
.cluster-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 8px;
    font-family: 'Sora', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.cb-rentan   { background: #e53935; color: white; }
.cb-transisi { background: #f9a825; color: #5d3a00; }
.cb-mandiri  { background: #2e7d32; color: white; }
.cluster-name {
    font-family: 'Sora', sans-serif;
    font-size: 14px;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 6px;
}
.cluster-desc { font-size: 12px; color: #5a5a7a; line-height: 1.6; margin-bottom: 10px; }
.cstat {
    display: inline-block;
    background: rgba(255,255,255,0.7);
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 500;
    color: #1a1a2e;
    margin: 2px;
}
.count-badge {
    display: inline-block;
    background: rgba(255,255,255,0.5);
    border-radius: 20px;
    padding: 2px 8px;
    font-size: 11px;
    color: #5a5a7a;
    margin-top: 8px;
}

/* ── Rekomendasi ── */
.rekom-item {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    padding: 10px 12px;
    border-radius: 8px;
    border: 1px solid #e2e2f0;
    background: white;
    font-size: 12.5px;
    color: #5a5a7a;
    line-height: 1.6;
    margin-bottom: 8px;
}
.rdot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; margin-top: 5px; }
.rekom-zone {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-family: 'Sora', sans-serif;
    margin: 10px 0 6px 0;
}

/* ── Temuan utama ── */
.finding-grid { display: flex; gap: 14px; flex-wrap: wrap; }
.finding-card {
    flex: 1;
    min-width: 200px;
    padding: 14px;
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
    font-size: 13px;
    line-height: 1.65;
    color: rgba(255,255,255,0.85);
}
.finding-card-header {
    font-size: 10px;
    opacity: 0.6;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'Sora', sans-serif;
    margin-bottom: 6px;
}

/* ── Tabel klaster ── */
.tbl-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 10px;
    font-family: 'Sora', sans-serif;
}

/* ── Chart container ── */
.chart-card {
    background: white;
    border: 1px solid #e2e2f0;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 1px 4px rgba(35,26,161,0.07);
}

/* ── Map legend ── */
.map-legend { display: flex; gap: 16px; margin-top: 8px; }
.ml-item { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #5a5a7a; }
.ml-dot { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }

/* Streamlit override */
div[data-testid="stVerticalBlock"] > div { gap: 0; }
.stSelectbox label, .stMultiSelect label {
    font-family: 'Sora', sans-serif !important;
    font-size: 11px !important;
    color: #5a5a7a !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
div[data-testid="stHorizontalBlock"] { gap: 16px; }

/* ── Container Grafik/Plotly ── */
div[data-testid="stPlotlyChart"] {
    background: white;
    border: 1px solid #e2e2f0;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 1px 4px rgba(35,26,161,0.07);
}
</style>
""", unsafe_allow_html=True)


# ── Load data (cached) ────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    df      = load_panel()
    ref     = load_ref()
    kl      = load_klaster()
    geojson = load_geojson()
    master  = build_master(df, kl, ref)
    corr_df = hitung_korelasi(df)
    return df, ref, kl, geojson, master, corr_df

df, ref, kl, geojson, master, corr_df = get_data()

TAHUN_LIST = sorted(df["tahun"].unique())
KAB_LIST   = sorted(master["nama_kabkota"].unique())
VAR_OPTIONS = {
    "Indeks Pembangunan Maysarakat": ("ipm", "IPM"),
    "Tingkat Pengangguran Terbuka (%)": ("tpt", "TPT (%)"),
    "Produk Domestik Regional Bruto (Miliar Rp)": ("pdrb", "PDRB (Miliar Rp)"),
    "Laju Pertumbuhan PDRB (%)": ("pertumbuhan_pdrb", "Pertumbuhan PDRB (%)"),
}


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="dash-header">
  <div>
    <div class="dash-eyebrow">Pusat Analisis Kebijakan &bull; Provinsi Aceh</div>
    <div class="dash-title">Peta Risiko &amp; Segmentasi Kemiskinan Aceh</div>
    <div class="dash-subtitle">Analisis berbasis K-Means Clustering &bull; Periode 2015–2025 &bull; 23 Kabupaten/Kota</div>
  </div>
  <div class="header-badges">
    <span class="hbadge">K-Means 3 Klaster</span>
    <span class="hbadge">Data BPS</span>
    <span class="hbadge">2015–2025</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FILTER BAR GLOBAL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
st.markdown('<div class="filter-label">&#9881; Filter Global</div>', unsafe_allow_html=True)

fc1, fc2, fc3, fc4 = st.columns([1.2, 2.5, 1.5, 2])
with fc1:
    tahun_filter = st.selectbox(
        "Tahun",
        options=["Semua"] + TAHUN_LIST,
        index=0,
        key="f_tahun",
    )
with fc2:
    wilayah_filter = st.multiselect(
        "Wilayah (untuk tren)",
        options=KAB_LIST,
        default=[],
        placeholder="Pilih kab/kota untuk dibandingkan…",
        key="f_wilayah",
    )
with fc3:
    var_key = st.selectbox(
        "Variabel Prediktor",
        options=list(VAR_OPTIONS.keys()),
        index=0,
        key="f_var",
    )
with fc4:
    klaster_filter = st.multiselect(
        "Filter Klaster",
        options=LABEL_KLASTER_ORDER,
        default=[],
        placeholder="Semua klaster…",
        key="f_klaster",
    )

# Active tags
var_field, var_label = VAR_OPTIONS[var_key]
tahun_tag   = str(tahun_filter) if tahun_filter != "Semua" else "2015–2025"
wilayah_tag = ", ".join(wilayah_filter) if wilayah_filter else "Semua Kab/Kota"
klaster_tag = ", ".join(klaster_filter) if klaster_filter else "Semua Klaster"

st.markdown(
    f'<div class="active-tags">'
    f'<span class="atag">📅 {tahun_tag}</span>'
    f'<span class="atag">📍 {wilayah_tag}</span>'
    f'<span class="atag">📊 {var_key}</span>'
    f'<span class="atag">🗂 {klaster_tag}</span>'
    f'</div>',
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)


# ── Terapkan filter ke dataframe ──────────────────────────────────────────────
if tahun_filter == "Semua":
    df_filt    = master.copy()
    tahun_kpi  = int(master["tahun"].max())
    tahun_peta = int(master["tahun"].max())
else:
    tahun_sel  = int(tahun_filter)
    df_filt    = master[master["tahun"] == tahun_sel].copy()
    tahun_kpi  = tahun_sel
    tahun_peta = tahun_sel

if klaster_filter:
    df_filt = df_filt[df_filt["nama_klaster"].isin(klaster_filter)]

# Data untuk tren: gunakan master penuh (agar sumbu X tetap lengkap)
df_tren = master.copy()
if klaster_filter:
    df_tren = df_tren[df_tren["nama_klaster"].isin(klaster_filter)]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">01 — Overview Kemiskinan Provinsi Aceh</div>', unsafe_allow_html=True)

# ── KPI Cards ────────────────────────────────────────────────────────────────
kpi = kpi_provinsi(master, tahun_kpi if tahun_filter != "Semua" else None)

def _delta_html(delta, invert=False):
    """Render badge delta. invert=True berarti kenaikan adalah buruk (mis. kemiskinan)."""
    if delta is None:
        return '<span class="kpi-delta-neu">— data</span>'
    good = (delta < 0) if invert else (delta > 0)
    arrow = "▼" if delta < 0 else "▲"
    cls   = "kpi-delta-up" if good else "kpi-delta-down"
    return f'<span class="{cls}">{arrow} {abs(delta):.2f}</span>'

n_rentan = int(
    master[master["tahun"] == tahun_kpi]["nama_klaster"]
    .value_counts()
    .get("Zona Rentan", 0)
)

kpi_cols = st.columns(6)
kpi_data = [
    ("Persentase Penduduk Miskin", kpi["pct_miskin"][0], "%",         _delta_html(kpi["pct_miskin"][1],       invert=True),  "kpi-navy"),
    ("Indeks Pembangunan Maysarakat (IPM)", kpi["ipm"][0], "", _delta_html(kpi["ipm"][1], invert=False), "kpi-green"),
    ("Tingkat Pengangguran Terbuka (TPT)", kpi["tpt"][0], "%", _delta_html(kpi["tpt"][1], invert=True), "kpi-yellow"),
    ("Produk Domestik Regional Bruto (PDRB) (Rp)", f"{kpi['pdrb'][0]:,.0f}".replace(",", "."), " Miliar", _delta_html(kpi["pertumbuhan_pdrb"][1], invert=False), "kpi-muted"),
    ("Laju Pertumbuhan PDRB", kpi["pertumbuhan_pdrb"][0], "%", _delta_html(kpi["pertumbuhan_pdrb"][1], invert=False), "kpi-muted"),
    ("ZONA RENTAN", n_rentan, " daerah",
     f'<span class="kpi-delta-neu">dari 23 kab/kota</span>', "kpi-red"),
]

for col, (label, val, unit, delta_html, cls) in zip(kpi_cols, kpi_data):
    with col:
        st.markdown(
            f'<div class="kpi-card {cls}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{val}<span class="kpi-unit">{unit}</span></div>'
            f'{delta_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Row: Multi-line tren + Top-5 Bar ─────────────────────────────────────────
row1_l, row1_r = st.columns([2, 1])

with row1_l:
    st.plotly_chart(
        fig_tren_kemiskinan(df_tren, wilayah_filter),
        use_container_width=True, config={"displayModeBar": False},
    )
    # Insight tren
    aceh_2015 = master[master["tahun"] == 2015]["pct_miskin"].mean()
    aceh_last  = master[master["tahun"] == master["tahun"].max()]["pct_miskin"].mean()
    delta_10yr = aceh_2015 - aceh_last
    st.markdown(
        f'<div class="insight-box">'
        f'<div class="insight-title"><span class="insight-dot">i</span> Insight Tren</div>'
        f'Kemiskinan Aceh turun dari <span class="hl">{aceh_2015:.2f}%</span> (2015) menjadi '
        f'<span class="hl">{aceh_last:.2f}%</span> ({master["tahun"].max()}), '
        f'penurunan kumulatif <span class="hl">{delta_10yr:.2f} pp</span> dalam 10 tahun. '
        f'Lonjakan 2020–2021 mencerminkan dampak pandemi yang memengaruhi seluruh wilayah secara bersamaan.'
        f'</div>',
        unsafe_allow_html=True,
    )

with row1_r:
    st.plotly_chart(
        fig_top5_termiskin(df_tren),
        use_container_width=True, config={"displayModeBar": False},
    )
    worst = df_tren.groupby("nama_kabkota")["pct_miskin"].mean().nlargest(1)
    best  = df_tren.groupby("nama_kabkota")["pct_miskin"].mean().nsmallest(1)
    st.markdown(
        f'<div class="insight-box">'
        f'<div class="insight-title"><span class="insight-dot">i</span> Temuan Utama</div>'
        f'<span class="hl">{worst.index[0]}</span> ({worst.iloc[0]:.2f}%) dan '
        f'<span class="hl">{best.index[0]}</span> ({best.iloc[0]:.2f}%) '
        f'mencerminkan disparitas yang sangat lebar — selisih hingga '
        f'<span class="hl">{(worst.iloc[0]-best.iloc[0]):.1f} pp</span> antar wilayah.'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 + 3 — TREN VARIABEL & KORELASI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">02 — Tren Variabel &amp; Analisis Korelasi</div>', unsafe_allow_html=True)

row2_1_l, row2_1_r = st.columns([2, 1])

with row2_1_l:
    st.plotly_chart(
        fig_tren_vs_variabel(df_tren, var_field, var_label),
        use_container_width=True, config={"displayModeBar": False},
    )

with row2_1_r:
    # Insight dinamis berdasarkan variabel
    r_val = corr_df[corr_df["field"] == var_field]["r"].values
    r_str = f"{r_val[0]:+.2f}" if len(r_val) else "—"
    insights_var = {
        "ipm": (
            f"IPM adalah prediktor terkuat kemiskinan di Aceh (r = <span class='hl'>{r_str}</span>). "
            "Peningkatan IPM yang konsisten berbanding terbalik dengan tren kemiskinan. "
            "Lonjakan kemiskinan 2020–2021 tidak diimbangi kenaikan IPM yang memadai."
        ),
        "tpt": (
            f"TPT dan kemiskinan bergerak searah (r = <span class='hl'>{r_str}</span>), terutama terlihat "
            "pada kenaikan simultan di 2020. Hubungannya lebih lemah dari IPM, mengindikasikan "
            "pekerjaan informal yang tidak tertangkap statistik TPT turut berperan."
        ),
        "pdrb": (
            f"Level PDRB berkorelasi negatif dengan kemiskinan (r = <span class='hl'>{r_str}</span>). "
            "Namun pertumbuhan tidak selalu merata — kabupaten dengan PDRB tinggi berbasis industri "
            "belum tentu memiliki kemiskinan rendah jika distribusinya tidak inklusif."
        ),
        "pertumbuhan_pdrb": (
            f"Pertumbuhan PDRB (r = <span class='hl'>{r_str}</span>) memiliki korelasi paling lemah, "
            "mengindikasikan <span class='hl'>pertumbuhan yang belum inklusif</span>. "
            "Pertumbuhan tinggi perlu disertai distribusi yang merata agar berdampak pada kemiskinan."
        ),
    }
    st.markdown(
        f'<div class="insight-box" style="margin-top: 0; height: 100%;">'
        f'<div class="insight-title"><span class="insight-dot">i</span> Insight Tren</div>'
        f'{insights_var.get(var_field, "—")}'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

row2_2_l, row2_2_r = st.columns(2)

with row2_2_l:
    # Scatter plot
    st.plotly_chart(
        fig_scatter_korelasi(df_tren, var_field, var_label),
        use_container_width=True, config={"displayModeBar": False},
    )

with row2_2_r:
    # Bar korelasi semua variabel
    st.plotly_chart(
        fig_bar_korelasi(corr_df),
        use_container_width=True, config={"displayModeBar": False},
    )

# Nilai korelasi variabel terpilih & Insight Korelasi
r_row = corr_df[corr_df["field"] == var_field]
if not r_row.empty:
    r = r_row.iloc[0]["r"]
    abs_r = abs(r)
    kuat = "Kuat" if abs_r >= 0.6 else "Sedang" if abs_r >= 0.4 else "Lemah"
    arah = "Negatif" if r < 0 else "Positif"
    badge_color = "#e53935" if abs_r >= 0.6 else "#f9a825" if abs_r >= 0.4 else "#2e7d32"
    badge_html = (
        f'<span style="background:{badge_color}15;color:{badge_color};'
        f'font-family:Sora,sans-serif;font-size:12px;font-weight:600;'
        f'padding:4px 10px;border-radius:20px;display:inline-block;margin-left:8px;">'
        f'r = {r:+.2f} &nbsp;|&nbsp; {arah} &nbsp;|&nbsp; {kuat}'
        f'</span>'
    )
    
    insight_korelasi = f'<div class="insight-box" style="margin-top: 10px;">'
    insight_korelasi += f'<div class="insight-title"><span class="insight-dot">i</span> Insight Korelasi {badge_html}</div>'
    insight_korelasi += f'Korelasi antara kemiskinan dan {var_label} menunjukkan hubungan yang <span class="hl">{kuat.lower()}</span> dan <span class="hl">{arah.lower()}</span>.'
    
    if var_field == "ipm":
        insight_korelasi += " Wilayah dengan IPM tinggi secara konsisten mencatat tingkat kemiskinan yang lebih rendah."
    elif var_field == "tpt":
        insight_korelasi += " Terdapat indikasi bahwa peningkatan pengangguran seringkali diikuti oleh peningkatan kemiskinan, meskipun tidak selalu linier sempurna."
    elif var_field == "pdrb":
        insight_korelasi += " Secara umum PDRB lebih tinggi sejalan dengan kemiskinan rendah, namun perlu dilihat tingkat inklusivitas pertumbuhannya."
    elif var_field == "pertumbuhan_pdrb":
        insight_korelasi += " Hubungan yang lemah ini patut diwaspadai, menandakan pertumbuhan ekonomi di beberapa daerah belum efektif menetes ke masyarakat bawah."
    
    insight_korelasi += '</div>'
    st.markdown(insight_korelasi, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CLUSTERING & SPATIAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">03 — Segmentasi Risiko &amp; Analisis Spasial</div>', unsafe_allow_html=True)

# ── Row: Peta + Profil Klaster ────────────────────────────────────────────────
row3_l, row3_r = st.columns([3, 2])

with row3_l:
    st.markdown("**Peta Klaster Risiko Kemiskinan Aceh**")
    st.caption(f"Segmentasi K-Means 3 klaster · Tahun {tahun_peta}")

    try:
        st.plotly_chart(
            fig_peta_klaster(master, geojson, tahun_peta),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    except Exception as e:
        st.warning(f"Peta tidak dapat ditampilkan: {e}")

    # Legenda peta
    st.markdown(
        '<div class="map-legend">'
        '<div class="ml-item"><div class="ml-dot" style="background:#e53935;"></div>Zona Rentan (Risiko Tinggi)</div>'
        '<div class="ml-item"><div class="ml-dot" style="background:#f9a825;"></div>Zona Transisi (Risiko Sedang)</div>'
        '<div class="ml-item"><div class="ml-dot" style="background:#2e7d32;"></div>Zona Mandiri (Risiko Rendah)</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Insight spasial dinamis
    rentan_list = (
        master[master["tahun"] == tahun_peta][master["nama_klaster"] == "Zona Rentan"]["nama_kabkota"]
        .unique().tolist()
    )
    mandiri_list = (
        master[master["tahun"] == tahun_peta][master["nama_klaster"] == "Zona Mandiri"]["nama_kabkota"]
        .unique().tolist()
    )
    st.markdown(
        f'<div class="insight-box" style="margin-top:10px;">'
        f'<div class="insight-title"><span class="insight-dot">i</span> Insight Spasial · {tahun_peta}</div>'
        f'Zona Rentan (<span class="hl">{len(rentan_list)} daerah</span>): '
        f'{", ".join(rentan_list[:5])}{"…" if len(rentan_list) > 5 else ""}. '
        f'Zona Mandiri (<span class="hl">{len(mandiri_list)} daerah</span>): '
        f'{", ".join(mandiri_list[:5])}{"…" if len(mandiri_list) > 5 else ""}. '
        f'Pola spasial menunjukkan korelasi antara aksesibilitas geografis dan tingkat kemiskinan.'
        f'</div>',
        unsafe_allow_html=True,
    )

with row3_r:
    # Profil 3 klaster
    st.markdown("**Profil Karakteristik Tiap Klaster**")
    st.caption(f"Rata-rata indikator per zona risiko · {tahun_peta}")

    # Hitung centroid per klaster untuk tahun terpilih
    kl_yr = kl[kl["tahun"] == tahun_peta]
    profil = kl_yr.groupby("nama_klaster").agg(
        n=("nama_kabkota", "count"),
        miskin=("centroid_miskin", "first"),
        ipm=("centroid_ipm", "first"),
        tpt=("centroid_tpt", "first"),
        growth=("centroid_pdrb_growth", "first"),
    ).reset_index()

    cluster_meta = {
        "Zona Rentan": {
            "css": "cluster-card-rentan", "badge": "cb-rentan",
            "desc": "Kemiskinan tinggi, IPM rendah, TPT tinggi. Umumnya wilayah terpencil dengan akses terbatas.",
        },
        "Zona Transisi": {
            "css": "cluster-card-transisi", "badge": "cb-transisi",
            "desc": "Kemiskinan mendekati rata-rata provinsi. IPM dan pertumbuhan menunjukkan tren positif namun belum stabil.",
        },
        "Zona Mandiri": {
            "css": "cluster-card-mandiri", "badge": "cb-mandiri",
            "desc": "Kemiskinan di bawah rata-rata, IPM tinggi, ekonomi tumbuh stabil.",
        },
    }

    for klaster in LABEL_KLASTER_ORDER:
        row_data = profil[profil["nama_klaster"] == klaster]
        if row_data.empty:
            continue
        row_data = row_data.iloc[0]
        meta = cluster_meta.get(klaster, {})
        st.markdown(
            f'<div class="cluster-card {meta["css"]}">'
            f'<span class="cluster-badge {meta["badge"]}">{klaster}</span>'
            f'<div class="cluster-name">{"Risiko Tinggi" if klaster=="Zona Rentan" else "Risiko Sedang" if klaster=="Zona Transisi" else "Risiko Rendah"}</div>'
            f'<div class="cluster-desc">{meta["desc"]}</div>'
            f'<span class="cstat">Miskin: <b>{row_data["miskin"]:.1f}%</b></span>'
            f'<span class="cstat">IPM: <b>{row_data["ipm"]:.1f}</b></span>'
            f'<span class="cstat">TPT: <b>{row_data["tpt"]:.1f}%</b></span>'
            f'<span class="cstat">Growth: <b>{row_data["growth"]:.1f}%</b></span>'
            f'<div><span class="count-badge">{int(row_data["n"])} kab/kota</span></div>'
            f'</div><br>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Row: Grafik profil + Pergerakan klaster ───────────────────────────────────
row4_l, row4_r = st.columns([3, 2])

with row4_l:
    st.plotly_chart(
        fig_profil_klaster(kl, tahun_peta),
        use_container_width=True, config={"displayModeBar": False},
    )

with row4_r:
    st.plotly_chart(
        fig_pergerakan_klaster(kl),
        use_container_width=True, config={"displayModeBar": False},
    )

st.markdown(
    f'<div class="insight-box" style="margin-top:20px;">'
    f'<div class="insight-title"><span class="insight-dot">i</span> Insight Pergerakan</div>'
    f'Distribusi klaster tidak bersifat statis — sejumlah kabupaten mengalami pergeseran zona '
    f'dari tahun ke tahun. Gunakan filter tahun untuk melacak kemajuan tiap daerah.'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row: Tabel detail + Rekomendasi ──────────────────────────────────────────
row5_l, row5_r = st.columns([3, 2])

with row5_l:
    st.markdown("**Detail Kab/Kota per Klaster**")
    st.caption(f"Data indikator dan hasil klaster · {tahun_peta}")

    tbl = (
        master[master["tahun"] == tahun_peta][[
            "nama_kabkota", "nama_klaster", "pct_miskin", "ipm", "tpt", "pertumbuhan_pdrb"
        ]]
        .drop_duplicates("nama_kabkota")
        .sort_values("pct_miskin", ascending=False)
        .reset_index(drop=True)
    )

    if klaster_filter:
        tbl = tbl[tbl["nama_klaster"].isin(klaster_filter)]

    def _style_tbl(row):
        color_map = {
            "Zona Rentan":   "background-color:#fff5f5; color:#1a1a2e",
            "Zona Transisi": "background-color:#fffdf0; color:#1a1a2e",
            "Zona Mandiri":  "background-color:#f0fff4; color:#1a1a2e",
        }
        return [color_map.get(row["Klaster"], "")] * len(row)

    tbl_display = tbl.rename(columns={
        "nama_kabkota":      "Kabupaten/Kota",
        "nama_klaster":      "Klaster",
        "pct_miskin":        "Miskin (%)",
        "ipm":               "IPM",
        "tpt":               "TPT (%)",
        "pertumbuhan_pdrb":  "Growth PDRB (%)",
    })

    st.dataframe(
        tbl_display.style
            .apply(_style_tbl, axis=1)
            .format({
                "Miskin (%)":       "{:.2f}",
                "IPM":              "{:.2f}",
                "TPT (%)":          "{:.2f}",
                "Growth PDRB (%)":  "{:.2f}",
            })
            .background_gradient(subset=["Miskin (%)"], cmap="RdYlGn_r", vmin=0, vmax=25),
        use_container_width=True,
        hide_index=True,
        height=350,
    )

with row5_r:
    st.markdown("**Rekomendasi Kebijakan**")
    st.caption("Berbasis profil klaster dan temuan analisis")

    rekoms = [
        {
            "zone": "ZONA RENTAN", "color": "#e53935",
            "items": [
                ("<b>Percepatan infrastruktur dasar</b>",
                 "Prioritaskan konektivitas jalan dan akses internet di daerah terpencil untuk membuka isolasi geografis."),
                ("<b>Penguatan jaring pengaman sosial</b>",
                 "Perkuat validasi data PKH dan BPNT; tingkatkan sasaran program untuk mengurangi exclusion error."),
            ]
        },
        {
            "zone": "ZONA TRANSISI", "color": "#f9a825",
            "items": [
                ("<b>Investasi SDM &amp; vokasi</b>",
                 "Perluas program pelatihan kerja berbasis potensi lokal untuk menurunkan TPT secara berkelanjutan."),
                ("<b>Diversifikasi ekonomi lokal</b>",
                 "Dukung UMKM berbasis komoditas unggulan daerah agar pertumbuhan PDRB lebih inklusif."),
            ]
        },
        {
            "zone": "ZONA MANDIRI", "color": "#2e7d32",
            "items": [
                ("<b>Model replikasi &amp; mentoring</b>",
                 "Jadikan kota-kota mandiri sebagai pusat pembelajaran kebijakan; dorong transfer kapasitas ke daerah tertinggal."),
            ]
        },
    ]

    for r in rekoms:
        st.markdown(
            f'<div class="rekom-zone" style="color:{r["color"]};">{r["zone"]}</div>',
            unsafe_allow_html=True,
        )
        for title, body in r["items"]:
            st.markdown(
                f'<div class="rekom-item">'
                f'<span class="rdot" style="background:{r["color"]};"></span>'
                f'<span>{title} — {body}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TEMUAN UTAMA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">04 — Ringkasan &amp; Temuan Utama</div>', unsafe_allow_html=True)

# Hitung temuan dinamis
worst_kab   = master.groupby("nama_kabkota")["pct_miskin"].mean().idxmax()
worst_val   = master.groupby("nama_kabkota")["pct_miskin"].mean().max()
best_kab    = master.groupby("nama_kabkota")["pct_miskin"].mean().idxmin()
best_val    = master.groupby("nama_kabkota")["pct_miskin"].mean().min()
disparitas  = worst_val - best_val
top_corr    = corr_df.sort_values("r").iloc[0]  # korelasi negatif terkuat
aceh_2015_v = master[master["tahun"] == 2015]["pct_miskin"].mean()
aceh_last_v = master[master["tahun"] == master["tahun"].max()]["pct_miskin"].mean()

dist_last = master[master["tahun"] == master["tahun"].max()]["nama_klaster"].value_counts()

st.markdown(f"""
<div style="background:#231aa1;border-radius:12px;padding:20px 24px;">
  <div style="font-family:'Sora',sans-serif;font-size:10px;font-weight:700;letter-spacing:1.5px;
              color:white;text-transform:uppercase;margin-bottom:14px;">
    Temuan Kunci Dashboard
  </div>
  <div class="finding-grid">
    <div class="finding-card" style="background:white; color:#231aa1; border-left:3px solid #e53935;">
      <div class="finding-card-header" style="color:#231aa1; opacity:0.8;">Disparitas Tertinggi</div>
      Selisih kemiskinan antara <b style="color:#e53935;">{worst_kab}</b> ({worst_val:.1f}%)
      dan <b style="color:#2e7d32;">{best_kab}</b> ({best_val:.1f}%) mencapai
      <b style="color:#e53935;">{disparitas:.1f} pp</b> — disparitas antarwilayah yang sangat dalam.
    </div>
    <div class="finding-card" style="background:white; color:#231aa1; border-left:3px solid #f9a825;">
      <div class="finding-card-header" style="color:#231aa1; opacity:0.8;">Prediktor Terkuat</div>
      <b style="color:#f57c00;">{top_corr["variabel"]}</b> memiliki korelasi
      <b style="color:#f57c00;">r = {top_corr["r"]:.2f}</b> terhadap kemiskinan.
      Investasi pada indikator ini diprioritaskan untuk dampak kebijakan terbesar.
    </div>
    <div class="finding-card" style="background:white; color:#231aa1; border-left:3px solid #2e7d32;">
      <div class="finding-card-header" style="color:#231aa1; opacity:0.8;">Tren 10 Tahun</div>
      Kemiskinan Aceh turun dari <b style="color:#2e7d32;">{aceh_2015_v:.2f}%</b> (2015)
      menjadi <b style="color:#2e7d32;">{aceh_last_v:.2f}%</b> ({master["tahun"].max()}),
      namun lonjakan 2020–2021 menunjukkan <b style="color:#f57c00;">kerentanan terhadap guncangan eksternal</b>.
    </div>
    <div class="finding-card" style="background:white; color:#231aa1; border-left:3px solid #00796b;">
      <div class="finding-card-header" style="color:#231aa1; opacity:0.8;">Distribusi Klaster ({master["tahun"].max()})</div>
      <b style="color:#e53935;">{dist_last.get("Zona Rentan", 0)} Zona Rentan</b>,
      <b style="color:#f57c00;">{dist_last.get("Zona Transisi", 0)} Zona Transisi</b>,
      <b style="color:#2e7d32;">{dist_last.get("Zona Mandiri", 0)} Zona Mandiri</b>
      dari 23 kab/kota. Lebih dari sepertiga wilayah masih membutuhkan intervensi prioritas.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div style="border-top:1px solid #e2e2f0;padding:14px 0;'
    'font-size:11px;color:#9393b0;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;">'
    '<span>Sumber data: BPS Provinsi Aceh &bull; Metode: K-Means Clustering (k=3) &bull; '
    'Variabel: Kemiskinan, IPM, TPT, PDRB, Pertumbuhan PDRB</span>'
    '<span>Dashboard Analisis Kemiskinan Aceh &bull; 2015–2025</span>'
    '</div>',
    unsafe_allow_html=True,
)
