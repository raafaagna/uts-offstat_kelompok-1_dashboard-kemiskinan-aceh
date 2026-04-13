# Dashboard Analisis & Segmentasi Risiko Kemiskinan Aceh

Dashboard interaktif berbasis Streamlit untuk analisis kemiskinan 23 kabupaten/kota di Provinsi Aceh, periode 2015–2025.

## Struktur Folder

```
dashboard_aceh/
├── app.py                          # Aplikasi utama Streamlit
├── data_loader.py                  # Modul load & preprocessing data
├── charts.py                       # Modul pembuat grafik Plotly
├── requirements.txt                # Dependensi Python
├── .streamlit/
│   └── config.toml                 # Konfigurasi tema Streamlit
└── data/
    ├── data_panel_aceh.csv
    ├── ref_kabkota.csv
    ├── hasil_clustering_zona_final.csv
    └── Kabupaten-Kota (Provinsi Aceh).geojson
```

## Cara Menjalankan

### 1. Persyaratan
- Python 3.10 atau lebih baru
- pip (package manager)

### 2. Install dependensi

```bash
pip install -r requirements.txt
```

### 3. Jalankan dashboard

```bash
streamlit run app.py
```

Dashboard akan terbuka otomatis di browser: `http://localhost:8501`

## Fitur Dashboard

| Section | Komponen |
|---------|----------|
| **01 Overview** | 6 KPI cards · Tren kemiskinan multi-line · Top-5 bar chart |
| **02 Tren & Korelasi** | Dual-axis tren vs variabel · Scatter plot · Bar korelasi semua variabel |
| **03 Clustering & Spasial** | Peta choropleth klaster · Profil 3 klaster · Grafik profil · Pergerakan klaster · Tabel detail · Rekomendasi kebijakan |
| **04 Temuan Utama** | Panel ringkasan 4 temuan kunci (dinamis) |

## Filter Global

Semua filter terletak di bagian paling atas dashboard dan memengaruhi seluruh komponen:

- **Tahun** — tampilkan data tahun tertentu atau semua periode
- **Wilayah** — pilih kab/kota untuk dibandingkan di grafik tren
- **Variabel Prediktor** — ganti variabel di grafik tren, scatter plot, dan nilai korelasi
- **Filter Klaster** — batasi tampilan berdasarkan zona klaster

## Catatan Teknis

- Peta choropleth menggunakan file GeoJSON dengan `CC_2` sebagai key penghubung ke `kode_kabkota`
- Nilai korelasi dihitung menggunakan Pearson dari `scipy.stats`
- Semua nilai KPI dan insight bersifat dinamis — otomatis menyesuaikan filter aktif
- Data di-cache menggunakan `@st.cache_data` untuk performa optimal
