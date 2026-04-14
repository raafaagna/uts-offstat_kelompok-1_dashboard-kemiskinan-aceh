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
    fig_peta_klaster, fig_profil_klaster, fig_pergerakan_klaster,
    fig_gap_analysis,
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
    overflow: hidden !important;
}
div[data-testid="stPlotlyChart"] iframe {
    overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load data (cached) ────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    df_raw  = load_panel()
    
    # Ekstrak data agregat Provinsi Aceh
    df_prov = df_raw[df_raw["nama_kabkota"] == "Provinsi Aceh"].copy()
    df      = df_raw[df_raw["nama_kabkota"] != "Provinsi Aceh"].copy()
    
    ref     = load_ref()
    kl      = load_klaster()
    geojson = load_geojson()
    master  = build_master(df, kl, ref)
    corr_df = hitung_korelasi(df)
    return df, df_prov, ref, kl, geojson, master, corr_df

df, df_provinsi, ref, kl, geojson, master, corr_df = get_data()

TAHUN_LIST = sorted(df["tahun"].unique())
KAB_LIST   = sorted(master["nama_kabkota"].unique())
VAR_OPTIONS = {
    "Indeks Pembangunan Masyarakat": ("ipm", "IPM"),
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

fc1, fc2, fc3 = st.columns([1.5, 3.5, 3])
with fc1:
    tahun_filter = st.selectbox(
        "Tahun",
        options=["Semua Tahun"] + TAHUN_LIST,
        index=0,
        key="f_tahun",
    )
with fc2:
    wilayah_filter = st.multiselect(
        "Wilayah (untuk tren)",
        options=KAB_LIST,
        default=[],
        key="f_wilayah",
    )
with fc3:
    var_key = st.multiselect(
        "Variabel Prediktor",
        options=list(VAR_OPTIONS.keys()),
        default=["Indeks Pembangunan Masyarakat"],
        key="f_var",
    )

if len(var_key) == 0:
    var_fields = [v[0] for v in VAR_OPTIONS.values()]
    var_labels = [v[1] for v in VAR_OPTIONS.values()]
    var_tag = "Semua Variabel"
else:
    var_fields = [VAR_OPTIONS[k][0] for k in var_key]
    var_labels = [VAR_OPTIONS[k][1] for k in var_key]
    var_tag = ", ".join(var_key)

# Active tags
tahun_tag   = str(tahun_filter)
wilayah_tag = ", ".join(wilayah_filter) if wilayah_filter else "Semua Wilayah"

st.markdown(
    f'<div class="active-tags">'
    f'<span class="atag">📅 {tahun_tag}</span>'
    f'<span class="atag">📍 {wilayah_tag}</span>'
    f'<span class="atag">📊 {var_tag}</span>'
    f'</div>',
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)


# ── Terapkan filter ke dataframe ──────────────────────────────────────────────
if tahun_filter == "Semua Tahun":
    tahun_sel = max(TAHUN_LIST)
    df_filt = master.copy()
else:
    tahun_sel  = int(tahun_filter)
    df_filt    = master[master["tahun"] == tahun_sel].copy()

tahun_kpi  = tahun_sel
tahun_peta = tahun_sel

# Data untuk tren: gunakan master penuh (agar sumbu X tetap lengkap)
df_tren = master.copy()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">01 — Overview Kemiskinan Provinsi Aceh</div>', unsafe_allow_html=True)

# ── KPI Cards ────────────────────────────────────────────────────────────────
kpi = kpi_provinsi(df_provinsi, tahun_kpi)

def _delta_html(delta, invert=False):
    """Render badge delta. invert=True berarti kenaikan adalah buruk (mis. kemiskinan)."""
    if delta is None:
        return '<span class="kpi-delta-neu">— data</span>'
    good = (delta < 0) if invert else (delta > 0)
    arrow = "▼" if delta < 0 else "▲"
    cls   = "kpi-delta-up" if good else "kpi-delta-down"
    return f'<span class="{cls}">{arrow} {abs(delta):.2f}</span>'

n_klaster = int(
    master[master["tahun"] == tahun_kpi]["nama_klaster"]
    .value_counts()
    .get("Zona Rentan", 0)
)
klaster_kpi_label = "ZONA RENTAN"
klaster_kpi_cls = "kpi-red"

kpi_cols = st.columns(6)
kpi_data = [
    ("Persentase Penduduk Miskin", kpi["pct_miskin"][0], "%",         _delta_html(kpi["pct_miskin"][1],       invert=True),  "kpi-navy"),
    ("Indeks Pembangunan Masyarakat (IPM)", kpi["ipm"][0], "", _delta_html(kpi["ipm"][1], invert=False), "kpi-green"),
    ("Tingkat Pengangguran Terbuka (TPT)", kpi["tpt"][0], "%", _delta_html(kpi["tpt"][1], invert=True), "kpi-yellow"),
    ("Produk Domestik Regional Bruto (PDRB) (Rp)", f"{kpi['pdrb'][0]:,.0f}".replace(",", "."), " Miliar", _delta_html(kpi["pertumbuhan_pdrb"][1], invert=False), "kpi-muted"),
    ("Laju Pertumbuhan PDRB", kpi["pertumbuhan_pdrb"][0], "%", _delta_html(kpi["pertumbuhan_pdrb"][1], invert=False), "kpi-muted"),
    (klaster_kpi_label, n_klaster, " daerah",
     f'<span class="kpi-delta-neu">dari 23 kab/kota</span>', klaster_kpi_cls),
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
        fig_tren_vs_variabel(df_provinsi, var_fields, var_labels),
        use_container_width=True, config={"displayModeBar": False},
    )
    # Insight dinamis berdasarkan variabel
    insight_text = ""
    implikasi_text = ""
    for vf in var_fields:
        insights_var = {
            "ipm": "<b>IPM</b>: Peningkatan IPM yang konsisten dari tahun ke tahun teramati sejalan dengan penurunan tingkat kemiskinan, walaupun terkadang terdapat periode perlambatan kemiskinan saat IPM meningkat.",
            "tpt": "<b>TPT</b>: Tren TPT dan kemiskinan menunjukkan arah pergerakan yang cenderung serupa dari tahun ke tahun.",
            "pdrb": "<b>PDRB</b>: Tren peningkatan PDRB dari waktu ke waktu konsisten sejalan dengan tren penurunan persentase penduduk miskin.",
            "pertumbuhan_pdrb": "<b>Pertumbuhan PDRB</b>: Fluktuasi laju pertumbuhan PDRB yang dinamis menunjukkan perbandingan yang lebih bervariasi dengan tren kemiskinan."
        }
        implikasi_var = {
            "ipm": "Pemerintah perlu <b>memperkuat investasi pada sektor pendidikan, kesehatan, dan daya beli masyarakat</b>, serta <b>mengoptimalkan program perlindungan sosial</b> untuk menjaga konsistensi penurunan kemiskinan, terutama saat terjadi guncangan ekonomi.",
            "tpt": "Pemerintah disarankan <b>memperluas penciptaan lapangan kerja formal</b> dan <b>program kewirausahaan padat karya</b> untuk menekan pengangguran yang tinggi seiring dengan tingginya kemiskinan.",
            "pdrb": "Pemerintah perlu <b>memastikan pertumbuhan PDRB disertai distribusi ekonomi yang inklusif</b> untuk menekan angka kemiskinan yang stabil.",
            "pertumbuhan_pdrb": "Pemerintah perlu <b>menjaga stabilitas dan kualitas pertumbuhan ekonomi</b> dengan <b>memperkuat sektor ekonomi akar rumput</b> agar peningkatan PDRB juga berdampak signifikan pada pengurangan kemiskinan."
        }
        insight_text += f"<li style='margin-bottom:4px;'>{insights_var.get(vf, '')}</li>"
        implikasi_text += f"<li style='margin-bottom:4px;'>{implikasi_var.get(vf, '')}</li>"

    st.markdown(
        f'<div class="insight-box">'
        f'<div class="insight-title"><span class="insight-dot">i</span> Insight Variabel Prediktor</div>'
        f'<ul style="margin:0; padding-left:20px; margin-bottom:8px;">{insight_text}</ul>'
        f'<div class="insight-title" style="margin-top:8px;"><span class="insight-dot">💡</span> Implikasi Kebijakan</div>'
        f'<ul style="margin:0; padding-left:20px;">{implikasi_text}</ul>'
        f'</div>',
        unsafe_allow_html=True,
    )

with row1_r:
    st.plotly_chart(
        fig_top5_termiskin(df_filt),
        use_container_width=True, config={"displayModeBar": False},
    )
    worst = df_filt.groupby("nama_kabkota")["pct_miskin"].mean().nlargest(1)
    best  = df_filt.groupby("nama_kabkota")["pct_miskin"].mean().nsmallest(1)
    st.markdown(
        f'<div class="insight-box">'
        f'<div class="insight-title"><span class="insight-dot">i</span> Temuan Utama</div>'
        f'<span class="hl">{worst.index[0]}</span> ({worst.iloc[0]:.2f}%) dan '
        f'<span class="hl">{best.index[0]}</span> ({best.iloc[0]:.2f}%) '
        f'mencerminkan disparitas yang sangat lebar — selisih hingga '
        f'<span class="hl">{(worst.iloc[0]-best.iloc[0]):.1f} pp</span> antar wilayah.'
        f'<div class="insight-title" style="margin-top:8px;"><span class="insight-dot">💡</span> Implikasi Kebijakan</div>'
        f'Pemerintah perlu <b>menerapkan intervensi yang lebih terfokus pada daerah-daerah prioritas</b> dengan pendekatan berbasis wilayah, seperti <b>penguatan ekonomi lokal, infrastruktur dasar, dan akses layanan publik</b>.'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ANALISIS TREN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">02 — Analisis Tren</div>', unsafe_allow_html=True)

if not wilayah_filter:
    st.caption("💡 **Tip:** Pilih kabupaten/kota pada _Filter Global_ di atas untuk menampilkan dan membandingkan tren daerah dengan rata-rata Provinsi Aceh.")

st.plotly_chart(
    fig_tren_kemiskinan(df_tren, df_provinsi, wilayah_filter),
    use_container_width=True, config={"displayModeBar": False},
)
# Insight tren
aceh_2015 = df_provinsi[df_provinsi["tahun"] == 2015]["pct_miskin"].values[0]
aceh_last  = df_provinsi[df_provinsi["tahun"] == df_provinsi["tahun"].max()]["pct_miskin"].values[0]
delta_10yr = aceh_2015 - aceh_last

if not wilayah_filter:
    implikasi_tren = "Pemerintah perlu <b>menjaga keberlanjutan program penanggulangan kemiskinan</b> yang sudah efektif tingkat provinsi sekaligus <b>memperkuat sistem perlindungan sosial yang adaptif terhadap krisis</b>, seperti bantuan sosial yang responsif dan dukungan pemulihan ekonomi, agar dampak guncangan seperti pandemi tidak kembali meningkatkan kemiskinan secara signifikan."
else:
    implikasi_tren = f"Pada wilayah {', '.join(wilayah_filter)}, pemerintah daerah perlu <b>mengkaji ketahanan program terhadap guncangan eksternal</b> (seperti pandemi), dan menjaga fokus pada <b>penguatan perlindungan sosial yang efektif dan adaptif</b> agar tren penurunan kemiskinan dapat dipertahankan."

st.markdown(
    f'<div class="insight-box">'
    f'<div class="insight-title"><span class="insight-dot">i</span> Insight Tren Wilayah</div>'
    f'Kemiskinan Aceh turun dari <span class="hl">{aceh_2015:.2f}%</span> (2015) menjadi '
    f'<span class="hl">{aceh_last:.2f}%</span> ({master["tahun"].max()}), '
    f'penurunan kumulatif <span class="hl">{delta_10yr:.2f} pp</span> dalam 10 tahun. '
    f'Lonjakan 2020–2021 mencerminkan dampak pandemi yang memengaruhi seluruh wilayah secara bersamaan.'
    f'<div class="insight-title" style="margin-top:8px;"><span class="insight-dot">💡</span> Implikasi Kebijakan</div>'
    f'{implikasi_tren}'
    f'</div>',
    unsafe_allow_html=True,
)



# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CLUSTERING & SPATIAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">03 — Segmentasi Risiko &amp; Analisis Spasial</div>', unsafe_allow_html=True)

# ── Row: Peta + Profil Klaster ────────────────────────────────────────────────
row3_l, row3_r = st.columns([3, 2])

with row3_l:
    st.markdown("**Peta Klaster Risiko Kemiskinan Aceh**")
    st.caption(f"Segmentasi K-Means 3 klaster (ditentukan berdasarkan analisis *silhouette score*) · Tahun {tahun_peta}")

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
    transisi_list = (
        master[master["tahun"] == tahun_peta][master["nama_klaster"] == "Zona Transisi"]["nama_kabkota"]
        .unique().tolist()
    )
    st.markdown(
        f'<div class="insight-box" style="margin-top:10px;">'
        f'<div class="insight-title"><span class="insight-dot">i</span> Insight Spasial · {tahun_peta}</div>'
        f'Distribusi kemiskinan secara spasial menunjukkan adanya pengelompokan wilayah ke dalam zona rentan (<span class="hl">{len(rentan_list)} daerah</span>), zona transisi (<span class="hl">{len(transisi_list)} daerah</span>), dan zona mandiri (<span class="hl">{len(mandiri_list)} daerah</span>).'
        f'<div class="insight-title" style="margin-top:8px;"><span class="insight-dot">💡</span> Implikasi Kebijakan</div>'
        f'Diperlukan <b>strategi alokasi sumber daya yang lebih asimetris</b>, dengan memberikan <b>prioritas utama pada perbaikan infrastruktur, pendidikan, dan kesehatan bagi Zona Rentan</b> untuk mempercepat pengurangan disparitas wilayah.'
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
            f'<span class="cstat">Pertumbuhan PDRB: <b>{row_data["growth"]:.1f}%</b></span>'
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
    f'<div class="insight-title" style="margin-top:8px;"><span class="insight-dot">💡</span> Implikasi Kebijakan</div>'
    f'Pemerintah tingkat provinsi perlu <b>mengadopsi mekanisme monitoring dan evaluasi tahunan</b> untuk mendampingi wilayah yang regresi (turun klaster) serta <b>mengapresiasi dan memfasilitasi replikasi kebijakan</b> dari wilayah yang mengalami pergerakan positif antar klaster.'
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

    def _style_tbl(row):
        color_map = {
            "Zona Rentan":   "background-color:#ffebeb; color:#950d0d; font-weight: 500",
            "Zona Transisi": "background-color:#fffce0; color:#8d6c00; font-weight: 500",
            "Zona Mandiri":  "background-color:#e8f8ec; color:#12612a; font-weight: 500",
        }
        return [color_map.get(row["Klaster"], "")] * len(row)

    tbl_display = tbl.rename(columns={
        "nama_kabkota":      "Kabupaten/Kota",
        "nama_klaster":      "Klaster",
        "pct_miskin":        "Miskin (%)",
        "ipm":               "IPM",
        "tpt":               "TPT (%)",
        "pertumbuhan_pdrb":  "Pertumbuhan PDRB (%)",
    })

    st.dataframe(
        tbl_display.style
            .apply(_style_tbl, axis=1)
            .format({
                "Miskin (%)":       "{:.2f}",
                "IPM":              "{:.2f}",
                "TPT (%)":          "{:.2f}",
                "Pertumbuhan PDRB (%)": "{:.2f}",
            }),
        use_container_width=True,
        hide_index=True,
        height=480,
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
# SECTION 4 — ANALISIS KESENJANGAN TARGET
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">04 — Analisis Kesenjangan Target</div>', unsafe_allow_html=True)

st.plotly_chart(
    fig_gap_analysis(df_provinsi),
    use_container_width=True, config={"displayModeBar": False},
)

# Hitung status on-track untuk Provinsi & Kab/Kota
last_year = df_provinsi["tahun"].max()
last_val = df_provinsi[df_provinsi["tahun"] == last_year]["pct_miskin"].values[0]

# Mapping target max tahun 2025 s.d 2030
target_dict_upper = {2025: 13.00, 2026: 12.00, 2027: 10.60, 2028: 9.00, 2029: 7.39, 2030: 6.39}
target_dict_lower = {2025: 12.00, 2026: 10.60, 2027: 9.00, 2028: 7.39, 2029: 6.39, 2030: 5.39}

upper_target = target_dict_upper.get(last_year, 13.00) # Default asumsi menuju 2025
lower_target = target_dict_lower.get(last_year, 12.00)

# Klasifikasi Kab/Kota
df_eval = master[master["tahun"] == last_year].copy()
def assign_status(x):
    if x <= lower_target: return "On-Track"
    elif x <= upper_target: return "Behind"
    else: return "Off-Track"

df_eval["Status_Gap"] = df_eval["pct_miskin"].apply(assign_status)

off_track_kab = df_eval[df_eval["Status_Gap"] == "Off-Track"]["nama_kabkota"].tolist()
behind_kab = df_eval[df_eval["Status_Gap"] == "Behind"]["nama_kabkota"].tolist()
on_track_kab = df_eval[df_eval["Status_Gap"] == "On-Track"]["nama_kabkota"].tolist()

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="kpi-card" style="border-top:3px solid #e53935;"><div class="kpi-label">OFF-TRACK (>{upper_target}%)</div><div class="kpi-value">{len(off_track_kab)} <span class="kpi-unit">daerah</span></div><div style="font-size:11px;color:#5a5a7a;margin-top:4px;">Prioritas Intervensi Mendesak</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi-card" style="border-top:3px solid #f9a825;"><div class="kpi-label">BEHIND ({lower_target}% - {upper_target}%)</div><div class="kpi-value">{len(behind_kab)} <span class="kpi-unit">daerah</span></div><div style="font-size:11px;color:#5a5a7a;margin-top:4px;">Perlu Perbaikan Program</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="kpi-card" style="border-top:3px solid #2e7d32;"><div class="kpi-label">ON-TRACK (<{lower_target}%)</div><div class="kpi-value">{len(on_track_kab)} <span class="kpi-unit">daerah</span></div><div style="font-size:11px;color:#5a5a7a;margin-top:4px;">Sesuai Target RPJMD</div></div>', unsafe_allow_html=True)

st.markdown(
    '<div style="font-size: 12px; color: #5a5a7a; margin-top: 8px;">'
    '<b>Keterangan Status:</b><br>'
    '🔴 <b>OFF-TRACK:</b> Apabila kemiskinan daerah > batas atas target.<br>'
    '🟡 <b>BEHIND:</b> Apabila kemiskinan daerah berada di antara batas bawah dan batas atas.<br>'
    '🟢 <b>ON-TRACK:</b> Apabila kemiskinan daerah < batas bawah.'
    '</div>', unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

off_track_str = ", ".join(off_track_kab)

if last_year in target_dict_upper:
    if last_val <= upper_target:
        status_text = f"Tingkat kemiskinan provinsi (<span class='hl'>{last_val:.2f}%</span>) berada di bawah batas atas (<span class='hl'>{upper_target:.2f}%</span>) untuk tahun {last_year}."
    else:
        status_text = f"Tingkat kemiskinan provinsi (<span class='hl'>{last_val:.2f}%</span>) berjarak cukup jauh dari batas atas target (<span class='hl'>{upper_target:.2f}%</span>) untuk tahun {last_year}."
else:
    status_text = f"Tingkat kemiskinan pada tahun {last_year} adalah <span class='hl'>{last_val:.2f}%</span>. Mengingat target RPJMD 2025 di rentang <span class='hl'>12.00-13.00%</span>, diperlukan akselerasi perbaikan."

st.markdown(
    f'<div class="insight-box">'
    f'<div class="insight-title"><span class="insight-dot">i</span> Insight Kesenjangan Target & Prioritas Daerah ({last_year})</div>'
    f'{status_text} Evaluasi ketat menunjukkan bahwa <span class="hl">{len(off_track_kab)} daerah</span> berada pada status <b>Off-Track</b>. Beberapa daerah prioritas urgen: <span style="color:#e53935;font-weight:600;">{off_track_str}</span>.'
    f'<div class="insight-title" style="margin-top:8px;"><span class="insight-dot">💡</span> Rekomendasi Kebijakan</div>'
    f'Pemerintah Provinsi harus <b>segera melakukan pendampingan intensif pada {len(off_track_kab)} daerah Off-Track</b>. Alokasi Dana Otonomi Khusus (Otsus) dan APBD harus diprioritaskan pada program pemberdayaan ekonomi langsung (seperti UMKM dan padat karya) di wilayah-wilayah tersebut.'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TEMUAN UTAMA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">05 — Ringkasan &amp; Temuan Utama</div>', unsafe_allow_html=True)

# Hitung temuan dinamis
worst_kab   = master.groupby("nama_kabkota")["pct_miskin"].mean().idxmax()
worst_val   = master.groupby("nama_kabkota")["pct_miskin"].mean().max()
best_kab    = master.groupby("nama_kabkota")["pct_miskin"].mean().idxmin()
best_val    = master.groupby("nama_kabkota")["pct_miskin"].mean().min()
disparitas  = worst_val - best_val
aceh_2015_v = df_provinsi[df_provinsi["tahun"] == 2015]["pct_miskin"].values[0]
aceh_last_v = df_provinsi[df_provinsi["tahun"] == df_provinsi["tahun"].max()]["pct_miskin"].values[0]

dist_last = master[master["tahun"] == master["tahun"].max()]["nama_klaster"].value_counts()

prioritas_kab = ", ".join(rentan_list[:3])
st.markdown(f"""
<div style="background:#231aa1;border-radius:12px;padding:20px 24px;">
  <div style="font-family:'Sora',sans-serif;font-size:10px;font-weight:700;letter-spacing:1.5px;
              color:white;text-transform:uppercase;margin-bottom:14px;">
    Temuan & Rekomendasi Kunci Dashboard
  </div>
  <div class="finding-grid">
    <div class="finding-card" style="background:white; color:#231aa1; border-left:3px solid #e53935;">
      <div class="finding-card-header" style="color:#231aa1; opacity:0.8;">Disparitas & Tren</div>
      Selisih tertinggi kemiskinan (<b style="color:#e53935;">{worst_kab}</b> vs <b style="color:#2e7d32;">{best_kab}</b>) mencapai
      <b style="color:#e53935;">{disparitas:.1f} pp</b>. Meskipun kemiskinan turun dari <b style="color:#2e7d32;">{aceh_2015_v:.2f}%</b> (2015)
      menjadi <b style="color:#2e7d32;">{aceh_last_v:.2f}%</b> ({master["tahun"].max()}),  lonjakan 2020–2021 menunjukkan <b style="color:#f57c00;">kerentanan tinggi terhadap guncangan ekonomi</b>.
    </div>
    <div class="finding-card" style="background:white; color:#231aa1; border-left:3px solid #00796b;">
      <div class="finding-card-header" style="color:#231aa1; opacity:0.8;">Distribusi Klaster ({master["tahun"].max()})</div>
      Terdapat <b style="color:#e53935;">{dist_last.get("Zona Rentan", 0)} Zona Rentan</b>,
      <b style="color:#f57c00;">{dist_last.get("Zona Transisi", 0)} Zona Transisi</b>, dan 
      <b style="color:#2e7d32;">{dist_last.get("Zona Mandiri", 0)} Zona Mandiri</b>.  Lebih dari sepertiga daerah wilayah provinsi Aceh masih membutuhkan intervensi prioritas.
    </div>
    <div class="finding-card" style="background:white; color:#231aa1; border-left:3px solid #8e24aa;">
      <div class="finding-card-header" style="color:#231aa1; opacity:0.8;">Prioritas & Rekomendasi</div>
      Pemerintah perlu memfokuskan <b>perbaikan infrastruktur dasar, pendidikan, dan kesehatan</b> pada daerah berisiko tinggi (Zona Rentan) seperti <b style="color:#e53935;">{prioritas_kab}</b>, serta <b>memperkuat jaring pengaman sosial yang adaptif</b> agar tren penurunan kemiskinan tetap konsisten.
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
    'Variabel: Persentase Penduduk Miskin, IPM, TPT, PDRB, Pertumbuhan PDRB</span>'
    '<span>Dashboard Analisis Kemiskinan Aceh &bull; 2015–2025</span>'
    '</div>',
    unsafe_allow_html=True,
)
