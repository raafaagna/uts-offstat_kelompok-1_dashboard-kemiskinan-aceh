"""
data_loader.py — Memuat dan mempersiapkan semua data dashboard.
Dipanggil sekali saat startup, hasil di-cache oleh Streamlit.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

DATA_DIR = Path(__file__).parent / "data"

# ── Mapping nama kab/kota antar file ──────────────────────────────────────────
# Panel CSV pakai huruf kecil + ejaan tidak baku → perlu dinormalisasi
_NAMA_MAP = {
    "simuele":        "Simeulue",
    "aceh singkil":   "Aceh Singkil",
    "aceh selatan":   "Aceh Selatan",
    "aceh tenggara":  "Aceh Tenggara",
    "aceh timur":     "Aceh Timur",
    "aceh tengah":    "Aceh Tengah",
    "aceh besar":     "Aceh Besar",
    "pidie":          "Pidie",
    "bireuen":        "Bireuen",
    "biruen":         "Bireuen",
    "aceh utara":     "Aceh Utara",
    "aceh barat daya":"Aceh Barat Daya",
    "gayo lues":      "Gayo Lues",
    "aceh tamiang":   "Aceh Tamiang",
    "nagan raya":     "Nagan Raya",
    "aceh jaya":      "Aceh Jaya",
    "bener meriah":   "Bener Meriah",
    "pidie jaya":     "Pidie Jaya",
    "kota banda aceh":"Kota Banda Aceh",
    "kota sabang":    "Kota Sabang",
    "kota langsa":    "Kota Langsa",
    "kota lhokseumawe":"Kota Lhokseumawe",
    "kota subulussam":"Kota Subulussalam",
    "kota subulussalam":"Kota Subulussalam",
    "aceh barat":     "Aceh Barat",
}

# Mapping nama_klaster → warna hex
WARNA_KLASTER = {
    "Zona Rentan":   "#e53935",
    "Zona Transisi": "#f9a825",
    "Zona Mandiri":  "#2e7d32",
}

LABEL_KLASTER_ORDER = ["Zona Rentan", "Zona Transisi", "Zona Mandiri"]

# ── Konstanta warna utama ─────────────────────────────────────────────────────
NAVY       = "#231aa1"
NAVY_LIGHT = "#3d31c4"
NAVY_PALE  = "#eceafc"
BG_COLOR   = "#f5f6fa"
TEXT_SEC   = "#5a5a7a"
BORDER     = "#e2e2f0"


def _norm_nama(s: str) -> str:
    return _NAMA_MAP.get(str(s).strip().lower(), str(s).strip().title())


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "data_panel_aceh.csv", sep=";", dtype_backend="numpy_nullable")
    df["nama_kabkota"] = df["nama_kabkota"].apply(_norm_nama)
    df["kode_kabkota"] = df["kode_kabkota"].astype(int)
    df["tahun"] = df["tahun"].astype(int)
    for col in ["pct_miskin", "ipm", "tpt", "pdrb", "pertumbuhan_pdrb"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_ref() -> pd.DataFrame:
    ref = pd.read_csv(DATA_DIR / "ref_kabkota.csv", sep=";")
    ref["kode_kabkota"] = ref["kode_kabkota"].astype(int)
    # Koordinat pakai koma desimal (Indonesian locale) → ganti ke titik
    for col in ["koordinat_lat", "koordinat_lon"]:
        ref[col] = ref[col].astype(str).str.replace(",", ".").astype(float)
    # Tambahkan nama_pendek jika belum ada
    if "nama_pendek" not in ref.columns:
        ref["nama_pendek"] = ref["nama_kabkota"]
    # Baris Provinsi Aceh tidak diperlukan
    ref = ref[ref["nama_kabkota"] != "Provinsi Aceh"].copy()
    return ref


def load_klaster() -> pd.DataFrame:
    kl = pd.read_csv(DATA_DIR / "hasil_clustering_zona_final.csv")
    kl["kode_kabkota"] = kl["kode_kabkota"].astype(int)
    kl["tahun"] = kl["tahun"].astype(int)
    return kl


def load_geojson() -> dict:
    with open(DATA_DIR / "Kabupaten-Kota (Provinsi Aceh).geojson", encoding="utf-8") as f:
        return json.load(f)


def build_master(df: pd.DataFrame, kl: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Join panel + klaster + ref menjadi satu tabel master."""
    master = df.merge(
        kl[["kode_kabkota", "tahun", "klaster_id", "nama_klaster",
            "centroid_miskin", "centroid_ipm", "centroid_tpt", "centroid_pdrb_growth"]],
        on=["kode_kabkota", "tahun"],
        how="left",
    )
    master = master.merge(
        ref[["kode_kabkota", "nama_pendek", "jenis_wilayah", "wilayah_adat",
             "koordinat_lat", "koordinat_lon"]],
        on="kode_kabkota",
        how="left",
    )
    return master


def hitung_korelasi(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung korelasi Pearson setiap prediktor terhadap pct_miskin."""
    rows = []
    for col, label in [
        ("ipm",              "IPM"),
        ("tpt",              "TPT (%)"),
        ("pdrb",             "PDRB (Miliar Rp)"),
        ("pertumbuhan_pdrb", "Pertumbuhan PDRB (%)"),
    ]:
        valid = df[["pct_miskin", col]].dropna()
        r, p = pearsonr(valid["pct_miskin"], valid[col])
        rows.append({"variabel": label, "field": col, "r": round(r, 3), "p": round(p, 4)})
    return pd.DataFrame(rows)


def kpi_provinsi(df: pd.DataFrame, tahun_sel: int | None = None) -> dict:
    """Ambil nilai KPI terakhir atau tahun tertentu, beserta delta YoY."""
    if tahun_sel:
        cur = df[df["tahun"] == tahun_sel]
        prev = df[df["tahun"] == tahun_sel - 1]
    else:
        max_y = df["tahun"].max()
        cur  = df[df["tahun"] == max_y]
        prev = df[df["tahun"] == max_y - 1]

    def mean_delta(col):
        c = cur[col].mean()
        p = prev[col].mean() if len(prev) else np.nan
        return round(c, 2), round(c - p, 2) if not np.isnan(p) else None

    return {
        "pct_miskin":       mean_delta("pct_miskin"),
        "ipm":              mean_delta("ipm"),
        "tpt":              mean_delta("tpt"),
        "pdrb":             mean_delta("pdrb"),
        "pertumbuhan_pdrb": mean_delta("pertumbuhan_pdrb"),
    }
