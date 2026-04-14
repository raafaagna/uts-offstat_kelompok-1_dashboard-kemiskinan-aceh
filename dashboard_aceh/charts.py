"""
charts.py — Semua fungsi pembuat grafik Plotly untuk dashboard.
Setiap fungsi mengembalikan plotly.graph_objects.Figure.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader import (
    NAVY, NAVY_PALE, BG_COLOR, TEXT_SEC, BORDER,
    WARNA_KLASTER, LABEL_KLASTER_ORDER,
)

# ── Tema dasar untuk semua grafik ────────────────────────────────────────────
_FONT_FAMILY = "DM Sans, Sora, sans-serif"

def _base_layout(**kwargs) -> dict:
    base = dict(
        font=dict(family=_FONT_FAMILY, color="#1a1a2e", size=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=12, r=12, t=36, b=12),
        height=400,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.22,
            xanchor="left", x=0,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor=BORDER, gridwidth=0.5, linecolor=BORDER,
            tickfont=dict(size=11), title_font=dict(size=11),
        ),
        yaxis=dict(
            gridcolor=BORDER, gridwidth=0.5, linecolor=BORDER,
            tickfont=dict(size=11), title_font=dict(size=11),
        ),
    )
    base.update(kwargs)
    return base


# ── 1. Multi-line tren kemiskinan ─────────────────────────────────────────────
def fig_tren_kemiskinan(df: pd.DataFrame, df_provinsi: pd.DataFrame, kab_terpilih: list[str]) -> go.Figure:
    """
    Garis tren kemiskinan: rata-rata Aceh + kab/kota yang dipilih.
    df sudah difilter berdasarkan tahun.
    """
    fig = go.Figure()
    years = sorted(df["tahun"].unique())

    # Rata-rata provinsi
    aceh_mean = df_provinsi.groupby("tahun")["pct_miskin"].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=aceh_mean["tahun"], y=aceh_mean["pct_miskin"].round(2),
        name="Provinsi Aceh",
        mode="lines+markers",
        line=dict(color=NAVY, width=3),
        marker=dict(size=6),
        hovertemplate="<b>Aceh (rata-rata)</b><br>Tahun: %{x}<br>Kemiskinan: %{y:.2f}%<extra></extra>",
    ))

    # Kab/kota terpilih
    palette = ["#e53935", "#f9a825", "#2e7d32", "#0288d1", "#7b1fa2",
               "#e64a19", "#00796b", "#c62828", "#1565c0", "#558b2f"]
    for i, kab in enumerate(kab_terpilih):
        sub = df[df["nama_kabkota"] == kab].sort_values("tahun")
        if sub.empty:
            continue
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=sub["tahun"], y=sub["pct_miskin"].round(2),
            name=kab,
            mode="lines+markers",
            line=dict(color=color, width=2, dash="dot"),
            marker=dict(size=5),
            hovertemplate=f"<b>{kab}</b><br>Tahun: %{{x}}<br>Kemiskinan: %{{y:.2f}}%<extra></extra>",
        ))

    fig.update_layout(
        **_base_layout(
            title=dict(text="Tren Kemiskinan: Provinsi vs Kabupaten/Kota", font=dict(size=13, color="#1a1a2e"), x=0, xanchor="left"),
            yaxis=dict(title="Kemiskinan (%)", ticksuffix="%", gridcolor=BORDER, gridwidth=0.5, linecolor=BORDER, tickfont=dict(size=11), title_font=dict(size=11)),
            xaxis=dict(title="Tahun", dtick=1, gridcolor=BORDER, gridwidth=0.5, linecolor=BORDER, tickfont=dict(size=11), title_font=dict(size=11)),
            hovermode="x unified",
        )
    )
    return fig


# ── 2. Bar chart Top-5 termiskin ──────────────────────────────────────────────
def fig_top5_termiskin(df: pd.DataFrame) -> go.Figure:
    top5 = (
        df.groupby("nama_kabkota")["pct_miskin"]
        .mean()
        .nlargest(5)
        .reset_index()
        .sort_values("pct_miskin")
    )
    colors = []
    for v in top5["pct_miskin"]:
        if v >= 18:
            colors.append("#e53935")
        elif v >= 14:
            colors.append("#f9a825")
        else:
            colors.append("#2e7d32")

    fig = go.Figure(go.Bar(
        x=top5["pct_miskin"].round(2),
        y=top5["nama_kabkota"],
        orientation="h",
        marker_color=colors,
        text=top5["pct_miskin"].round(2).astype(str) + "%",
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Kemiskinan: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(
            title=dict(text="Top 5 Kabupaten/Kota Termiskin", font=dict(size=13, color="#1a1a2e"), x=0, xanchor="left"),
            xaxis=dict(title="Rata-rata Kemiskinan (%)", range=[0, top5["pct_miskin"].max() * 1.2], ticksuffix="%", gridcolor=BORDER, gridwidth=0.5, linecolor=BORDER, tickfont=dict(size=11), title_font=dict(size=11)),
            yaxis=dict(gridcolor=BORDER, gridwidth=0.5, linecolor=BORDER, tickfont=dict(size=11), title_font=dict(size=11)),
            margin=dict(l=12, r=60, t=36, b=12),
            showlegend=False,
        )
    )
    return fig


# ── 3. Tren kemiskinan vs variabel prediktor (dual axis) ─────────────────────
def fig_tren_vs_variabel(df: pd.DataFrame, var_fields: list[str], var_labels: list[str]) -> go.Figure:
    """Tren kemiskinan (kiri) vs variabel prediktor (kanan) rata-rata provinsi."""
    agg = df.groupby("tahun")[["pct_miskin"] + var_fields].mean().reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=agg["tahun"], y=agg["pct_miskin"].round(2),
        name="Kemiskinan (%)",
        mode="lines+markers",
        line=dict(color=NAVY, width=2.5),
        marker=dict(size=6),
        hovertemplate="Kemiskinan: %{y:.2f}%<extra></extra>",
    ), secondary_y=False)

    var_color_map = {
        "ipm":              "#2e7d32",
        "tpt":              "#f9a825",
        "pdrb":             "#231aa1",
        "pertumbuhan_pdrb": "#e53935",
    }

    for v_field, v_label in zip(var_fields, var_labels):
        var_color = var_color_map.get(v_field, "#555")
        fig.add_trace(go.Scatter(
            x=agg["tahun"], y=agg[v_field].round(2),
            name=v_label,
            mode="lines+markers",
            line=dict(color=var_color, width=2, dash="dash"),
            marker=dict(size=5),
            hovertemplate=f"{v_label}: %{{y:.2f}}<extra></extra>",
        ), secondary_y=True)

    y2_title = var_labels[0] if len(var_labels) == 1 else "Nilai Prediktor"
    
    fig.update_layout(
        **_base_layout(
            title=dict(text="Tren Persentase Penduduk Miskin vs Variabel Prediktor", font=dict(size=13, color="#1a1a2e"), x=0, xanchor="left"),
            hovermode="x unified",
            xaxis=dict(title="Tahun", dtick=1, gridcolor=BORDER, gridwidth=0.5, linecolor=BORDER, tickfont=dict(size=11), title_font=dict(size=11)),
        )
    )
    fig.update_yaxes(title_text="Kemiskinan (%)", ticksuffix="%",
                     gridcolor=BORDER, tickfont=dict(size=11), title_font=dict(size=11, color=NAVY),
                     secondary_y=False)
    fig.update_yaxes(title_text=y2_title,
                     gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11), title_font=dict(size=11),
                     secondary_y=True)
    return fig





# ── 6. Peta choropleth klaster ────────────────────────────────────────────────
def fig_peta_klaster(master: pd.DataFrame, geojson: dict, tahun: int) -> go.Figure:
    """Choropleth peta klaster per tahun menggunakan Plotly."""
    df_map = master[master["tahun"] == tahun][
        ["kode_kabkota", "nama_kabkota", "nama_klaster", "pct_miskin", "ipm", "tpt"]
    ].drop_duplicates("kode_kabkota")

    # Map klaster ke angka untuk colorscale
    klaster_num = {"Zona Rentan": 1, "Zona Transisi": 2, "Zona Mandiri": 3}
    df_map = df_map.copy()
    df_map["klaster_num"] = df_map["nama_klaster"].map(klaster_num).fillna(0)

    # Custom colorscale: merah – kuning – hijau
    colorscale = [
        [0.0,  "#e53935"],
        [0.5,  "#f9a825"],
        [1.0,  "#2e7d32"],
    ]

    fig = px.choropleth(
        df_map,
        geojson=geojson,
        locations="kode_kabkota",
        featureidkey="properties.CC_2",
        color="klaster_num",
        color_continuous_scale=colorscale,
        range_color=[1, 3],
        custom_data=["nama_kabkota", "nama_klaster", "pct_miskin", "ipm", "tpt"],
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Klaster: %{customdata[1]}<br>"
            "Kemiskinan: %{customdata[2]:.2f}%<br>"
            "IPM: %{customdata[3]:.2f}<br>"
            "TPT: %{customdata[4]:.2f}%"
            "<extra></extra>"
        ),
        marker_line_color="white",
        marker_line_width=1,
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="white",
    )

    fig.update_layout(
        paper_bgcolor="white",
        geo=dict(bgcolor="white"),
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_showscale=False,
        height=400,
    )
    return fig


# ── 7. Radar/bar profil klaster ───────────────────────────────────────────────
def fig_profil_klaster(kl: pd.DataFrame, tahun: int) -> go.Figure:
    """Grouped bar: perbandingan indikator rata-rata per klaster."""
    df_yr = kl[kl["tahun"] == tahun]
    agg = df_yr.groupby("nama_klaster").agg(
        miskin=("centroid_miskin", "first"),
        ipm=("centroid_ipm", "first"),
        tpt=("centroid_tpt", "first"),
        growth=("centroid_pdrb_growth", "first"),
    ).reset_index()

    # Urutkan
    agg["_order"] = agg["nama_klaster"].map({k: i for i, k in enumerate(LABEL_KLASTER_ORDER)})
    agg = agg.sort_values("_order")

    fig = go.Figure()
    metrics = [
        ("miskin",  "Kemiskinan (%)",      "%"),
        ("ipm",     "IPM",                 ""),
        ("tpt",     "TPT (%)",             "%"),
        ("growth",  "Pertumbuhan PDRB (%)","%"),
    ]

    for col, label, suffix in metrics:
        fig.add_trace(go.Bar(
            name=label,
            x=agg["nama_klaster"],
            y=agg[col].round(2),
            text=agg[col].round(2).astype(str) + suffix,
            textposition="outside",
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.2f}}{suffix}<extra></extra>",
        ))

    fig.update_layout(
        **_base_layout(
            title=dict(text="Profil Rata-rata per Klaster", font=dict(size=13, color="#1a1a2e"), x=0, xanchor="left"),
            barmode="group",
            xaxis=dict(gridcolor=BORDER, gridwidth=0.5, linecolor=BORDER, tickfont=dict(size=11), title_font=dict(size=11)),
            yaxis=dict(gridcolor=BORDER, gridwidth=0.5, linecolor=BORDER, tickfont=dict(size=11), title_font=dict(size=11)),
            colorway=["#e53935", "#2e7d32", "#f9a825", NAVY],
        )
    )
    return fig


# ── 8. Pergerakan klaster antar tahun (Sankey / area) ────────────────────────
def fig_pergerakan_klaster(kl: pd.DataFrame) -> go.Figure:
    """Area stacked: jumlah kab/kota per klaster per tahun."""
    cnt = kl.groupby(["tahun", "nama_klaster"]).size().reset_index(name="n")

    fig = go.Figure()
    colors = {"Zona Rentan": "#e53935", "Zona Transisi": "#f9a825", "Zona Mandiri": "#2e7d32"}
    for klaster in ["Zona Mandiri", "Zona Transisi", "Zona Rentan"]:
        sub = cnt[cnt["nama_klaster"] == klaster].sort_values("tahun")
        fig.add_trace(go.Scatter(
            x=sub["tahun"], y=sub["n"],
            name=klaster,
            mode="lines+markers",
            line=dict(color=colors[klaster], width=2),
            marker=dict(size=5),
            hovertemplate=f"<b>{klaster}</b><br>Tahun: %{{x}}<br>Jumlah: %{{y}} daerah<extra></extra>",
        ))

    fig.update_layout(
        **_base_layout(
            title=dict(text="Distribusi Klaster per Tahun (Jumlah Kab/Kota)", font=dict(size=13, color="#1a1a2e"), x=0, xanchor="left"),
            yaxis=dict(title="Jumlah Kab/Kota", dtick=1, gridcolor=BORDER, gridwidth=0.5, linecolor=BORDER, tickfont=dict(size=11), title_font=dict(size=11)),
            xaxis=dict(title="Tahun", dtick=1, gridcolor=BORDER, gridwidth=0.5, linecolor=BORDER, tickfont=dict(size=11), title_font=dict(size=11), range=[cnt["tahun"].min() - 0.5, cnt["tahun"].max() + 0.6]),
            hovermode="x unified",
            margin=dict(l=12, r=32, t=36, b=12),
        )
    )
    return fig


# ── 9. Analisis Kesenjangan Target ────────────────────────────
def fig_gap_analysis(df_provinsi: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # Rata-rata provinsi (historis)
    aceh_mean = df_provinsi.groupby("tahun")["pct_miskin"].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=aceh_mean["tahun"], y=aceh_mean["pct_miskin"].round(2),
        name="Realisasi Aceh",
        mode="lines+markers",
        line=dict(color=NAVY, width=3),
        marker=dict(size=6),
        hovertemplate="<b>Realisasi (Aceh)</b><br>Tahun: %{x}<br>Kemiskinan: %{y:.2f}%<extra></extra>",
    ))

    # Target RPJMD
    target_years = [2025, 2026, 2027, 2028, 2029, 2030]
    target_upper = [13.00, 12.00, 10.60, 9.00, 7.39, 6.39]
    target_lower = [12.00, 10.60, 9.00, 7.39, 6.39, 5.39]

    # Upper bound
    fig.add_trace(go.Scatter(
        x=target_years, y=target_upper,
        name="Batas Atas Target",
        mode="lines",
        line=dict(color="rgba(46, 125, 50, 0)", width=0),
        showlegend=False,
        hoverinfo="skip"
    ))

    # Lower bound (Fill)
    fig.add_trace(go.Scatter(
        x=target_years, y=target_lower,
        name="Target RPJMD",
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(46, 125, 50, 0.2)",
        line=dict(color="#2e7d32", width=2, dash="dash"),
        hovertemplate="<b>Target RPJMD</b><br>Tahun: %{x}<br>Range Target: %{y:.2f}% - %{customdata:.2f}%<extra></extra>",
        customdata=target_upper
    ))

    # Trace untuk batas atas agar garisnya terlihat (optional)
    fig.add_trace(go.Scatter(
        x=target_years, y=target_upper,
        name="Batas Atas (Target)",
        mode="lines",
        line=dict(color="#2e7d32", width=2, dash="dash"),
        showlegend=False,
        hoverinfo="skip"
    ))

    fig.update_layout(
        **_base_layout(
            title=dict(text="Analisis Kesenjangan Realisasi vs Target RPJMD", font=dict(size=13, color="#1a1a2e"), x=0, xanchor="left"),
            yaxis=dict(title="Kemiskinan (%)", ticksuffix="%", gridcolor=BORDER, gridwidth=0.5, linecolor=BORDER, tickfont=dict(size=11), title_font=dict(size=11)),
            xaxis=dict(title="Tahun", dtick=1, gridcolor=BORDER, gridwidth=0.5, linecolor=BORDER, tickfont=dict(size=11), title_font=dict(size=11)),
            hovermode="x unified",
        )
    )
    return fig

