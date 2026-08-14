import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from database.query import get_hasil_clustering, get_all_dataset

# Keamanan login. Jika user masuk tanpa login
if "login" not in st.session_state or not st.session_state.login:
    st.warning("Silahkan Login Terlebih Dahulu.")
    st.stop()

# Konten utama dashboard
st.title("DASHBOARD", anchor=False)
st.caption("Sistem Clustering Produktivitas Hasil Pertanian")
st.markdown("---")

# Ambil data dari database
try:
    data = get_hasil_clustering()
    raw_data = get_all_dataset()
except Exception as e:
    st.error(f"Gagal memuat data dari database. Error: {e}")
    st.stop()

if not data or len(data) == 0:
    st.warning("Belum ada data clustering yang tersedia. Silahkan admin untuk melakukan proses clustering terlebih dahulu.")
    st.stop()

df = pd.DataFrame(data)

# Data Mentah (Untuk nilai produktivitas asli dalam Ku/Ha)
if raw_data and len(raw_data) > 0:
    df_raw = pd.DataFrame(raw_data)
    df_raw.columns = [str(col).upper().strip() for col in df_raw.columns]
    df_raw["PRODUKTIVITAS"] = pd.to_numeric(df_raw["PRODUKTIVITAS"], errors="coerce").fillna(0.0)
    df_raw["TAHUN"] = pd.to_numeric(df_raw["TAHUN"], errors="coerce")
else:
    df_raw = pd.DataFrame()

# STANDARDISASI NAMA KOLOM (Huruf Kapital & Tanpa Spasi)
df.columns = [str(col).upper().strip() for col in df.columns]

# VALIDASI WAJIB: Cek apakah kolom CLUSTER ada di DataFrame
if "CLUSTER" not in df.columns:
    st.warning("⚠️ Kolom 'CLUSTER' belum ditemukan pada dataset. Pastikan Anda sudah menjalankan pemodelan di halaman **Clustering HDBSCAN**.")
    st.stop()

# Konversi kolom CLUSTER ke numerik secara aman
df["CLUSTER"] = pd.to_numeric(df["CLUSTER"], errors="coerce").fillna(-1).astype(int)

# -----------------------------------------------------------------------------
# 1. DETEKSI FITUR KOMODITAS ASLI & FITUR CLUSTERING
# -----------------------------------------------------------------------------
kolom_non_fitur = [
    'ID', 'ID_PREPROCESSING', 'TAHUN', 'KECAMATAN', 'CLUSTER', 
    'PROBABILITY', 'OUTLIER_SCORE', 'CLUSTER_FINAL', 'KATEGORI', 
    'LATITUDE', 'LONGITUDE', 'CREATED_AT', 'UPDATED_AT'
]

# Kolom komoditas murni (Bukan kolom Z-Score dan bukan metadata) -> 8 Komoditas
komoditas_cols = [
    c for c in df.columns 
    if c.upper() not in kolom_non_fitur and not c.upper().startswith('Z_')
]

# Kolom Z-Score (jika ada di database hasil clustering)
zscore_cols = [c for c in df.columns if c.upper().startswith('Z_')]
clustering_feature_cols = zscore_cols if len(zscore_cols) > 0 else komoditas_cols

# Pastikan tipe data numerik
for col in komoditas_cols + zscore_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

# -----------------------------------------------------------------------------
# 2. LOGIKA PEMETAAN KATEGORI CLUSTER (Tinggi, Sedang, Rendah, Noise)
# -----------------------------------------------------------------------------
df_valid = df[df["CLUSTER"] != -1].copy()

if len(df_valid) > 0:
    # Urutkan klaster berdasarkan rata-rata produktivitas komoditas secara descending
    if komoditas_cols:
        cluster_means = df_valid.groupby("CLUSTER")[komoditas_cols].mean().mean(axis=1).sort_values(ascending=False)
    elif "PRODUKTIVITAS" in df_valid.columns:
        cluster_means = df_valid.groupby("CLUSTER")["PRODUKTIVITAS"].mean().sort_values(ascending=False)
    else:
        cluster_means = df_valid.groupby("CLUSTER").size().sort_values(ascending=False)

    n_klaster_found = len(cluster_means)
    
    orig_to_new = {}
    orig_to_kategori = {}

    for idx, orig_id in enumerate(cluster_means.index):
        if n_klaster_found >= 3:
            if idx == 0:
                new_id = 0
                kat = "Cluster 0 (Tinggi)"
            elif idx == n_klaster_found - 1:
                new_id = 2
                kat = "Cluster 2 (Rendah)"
            else:
                new_id = 1
                kat = f"Cluster {idx} (Sedang)"
        elif n_klaster_found == 2:
            if idx == 0:
                new_id = 0
                kat = "Cluster 0 (Tinggi)"
            else:
                new_id = 2
                kat = "Cluster 2 (Rendah)"
        else:
            new_id = 1
            kat = "Cluster 1 (Sedang)"

        orig_to_new[orig_id] = new_id
        orig_to_kategori[orig_id] = kat

    orig_to_new[-1] = -1
    orig_to_kategori[-1] = "Noise (-1)"

    df["CLUSTER_FINAL"] = df["CLUSTER"].map(orig_to_new).fillna(-1).astype(int)
    df["KATEGORI"] = df["CLUSTER"].map(orig_to_kategori).fillna("Noise (-1)")
else:
    df["CLUSTER_FINAL"] = -1
    df["KATEGORI"] = "Noise (-1)"

color_discrete_map = {
    "Cluster 0 (Tinggi)": "#2CA02C",  # Hijau
    "Cluster 1 (Sedang)": "#1F77B4",  # Biru
    "Cluster 2 (Rendah)": "#FFBB11",  # Kuning/Oren
    "Noise (-1)": "#D62728"           # Merah
}
category_orders = ["Cluster 0 (Tinggi)", "Cluster 1 (Sedang)", "Cluster 2 (Rendah)", "Noise (-1)"]

# -----------------------------------------------------------------------------
# 3. FILTER TAHUN DATASET & METRIK RINGKASAN (KPI METRICS)
# -----------------------------------------------------------------------------
# Cek ketersediaan tahun pada data
if "TAHUN" in df.columns and df["TAHUN"].dropna().any():
    daftar_tahun = sorted(df["TAHUN"].dropna().unique().tolist(), reverse=True)
    
    col_filter1, _ = st.columns([1, 3])
    with col_filter1:
        opsi_tahun = ["Semua Tahun"] + [str(int(t)) for t in daftar_tahun]
        # Inisialisasi default dari session_state jika ada
        default_index = 0
        if "selected_tahun" in st.session_state and st.session_state["selected_tahun"] is not None:
            tahun_str = str(int(st.session_state["selected_tahun"]))
            if tahun_str in opsi_tahun:
                default_index = opsi_tahun.index(tahun_str)
                
        tahun_terpilih = st.selectbox("📅 Filter Tahun Dataset:", options=opsi_tahun, index=default_index)
        
    if tahun_terpilih != "Semua Tahun":
        df_view = df[df["TAHUN"] == int(tahun_terpilih)].copy()
        df_raw_view = df_raw[df_raw["TAHUN"] == int(tahun_terpilih)].copy() if not df_raw.empty else pd.DataFrame()
        label_tahun = f"Tahun {tahun_terpilih}"
    else:
        df_view = df.copy()
        df_raw_view = df_raw.copy()
        min_t, max_t = int(min(daftar_tahun)), int(max(daftar_tahun))
        label_tahun = f"{min_t} - {max_t}" if min_t != max_t else f"Tahun {min_t}"
else:
    df_view = df.copy()
    df_raw_view = df_raw.copy()
    label_tahun = "Tahun -"

# Hitung Metrik Berdasarkan Data Tampilan (df_view / df_raw_view)
m1, m2, m3, m4 = st.columns(4)
total_kecamatan = df_view["KECAMATAN"].nunique() if "KECAMATAN" in df_view.columns else len(df_view)

# Hitung Komoditas & Produktivitas Asli dari Raw Data
if not df_raw_view.empty and "PRODUKTIVITAS" in df_raw_view.columns:
    total_komoditas = df_raw_view["KOMODITAS"].nunique() if "KOMODITAS" in df_raw_view.columns else len(komoditas_cols)
    avg_prod = df_raw_view["PRODUKTIVITAS"].mean()
elif komoditas_cols:
    total_komoditas = len(komoditas_cols)
    avg_prod = df_view[komoditas_cols].values.mean()
else:
    total_komoditas = 0
    avg_prod = 0.0

m1.metric("Periode Dataset", f"{label_tahun}")
m2.metric("Total Kecamatan", f"{total_kecamatan} Kecamatan")
m3.metric("Total Komoditas Terdaftar", f"{total_komoditas} Komoditas")
m4.metric("Rata-rata Produktivitas", f"{avg_prod:.2f} Ku/Ha")

st.markdown("---")

# -----------------------------------------------------------------------------
# 4. GRAFIK & PETA SEBARAN CLUSTER
# -----------------------------------------------------------------------------
col_graph1, col_graph2 = st.columns([1, 1])

# KIRI: Grafik Produktivitas Asli
with col_graph1:
    st.markdown(f"### Rata-rata Produktivitas per Komoditas ({label_tahun})")
    
    if not df_raw_view.empty and "KOMODITAS" in df_raw_view.columns and "PRODUKTIVITAS" in df_raw_view.columns:
        df_komoditas = df_raw_view.groupby("KOMODITAS")["PRODUKTIVITAS"].mean().reset_index()
        df_komoditas["KOMODITAS"] = df_komoditas["KOMODITAS"].str.replace('_', ' ').str.title()
        
        fig_bar = px.bar(
            df_komoditas, x="KOMODITAS", y="PRODUKTIVITAS", text="PRODUKTIVITAS",
            color_discrete_sequence=["#1F77B4"] 
        )
        fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_bar.update_yaxes(title_text="Produktivitas (Ku/Ha)")
        fig_bar.update_xaxes(title_text="")
    elif komoditas_cols:
        # Fallback jika raw data tidak tersedia
        df_komoditas_pivot = df_view[komoditas_cols].mean().reset_index()
        df_komoditas_pivot.columns = ["KOMODITAS", "PRODUKTIVITAS"]
        df_komoditas_pivot["KOMODITAS"] = df_komoditas_pivot["KOMODITAS"].str.replace('_', ' ').str.title()
        
        fig_bar = px.bar(
            df_komoditas_pivot, x="KOMODITAS", y="PRODUKTIVITAS", text="PRODUKTIVITAS",
            color_discrete_sequence=["#1F77B4"]
        )
        fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_bar.update_yaxes(title_text="Produktivitas (Ku/Ha)")
        fig_bar.update_xaxes(title_text="")
    else:
        fig_bar = px.bar()

    fig_bar.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_bar, width="stretch", config={'displayModeBar': False})

# KANAN: Peta / Scatter Sebaran Cluster
with col_graph2:
    st.markdown(f"### Sebaran Cluster Kecamatan ({label_tahun})")
    if "LATITUDE" in df_view.columns and "LONGITUDE" in df_view.columns:
        fig_map = px.scatter_mapbox(
            df_view, lat="LATITUDE", lon="LONGITUDE", color="KATEGORI",
            color_discrete_map=color_discrete_map, hover_name="KECAMATAN",
            zoom=9, height=400, category_orders={"KATEGORI": category_orders}
        )
        fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        # Scatter Sebaran Cluster per Kecamatan
        fig_scatter_spatial = px.scatter(
            df_view, x="KECAMATAN", y="KATEGORI", color="KATEGORI",
            color_discrete_map=color_discrete_map,
            category_orders={"KATEGORI": category_orders},
            title=f"Sebaran Cluster per Kecamatan ({label_tahun})"
        )
        fig_scatter_spatial.update_layout(
            height=400, 
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis_title="Kategori Klaster",
            xaxis_title="Kecamatan"
        )
        st.plotly_chart(fig_scatter_spatial, width="stretch", config={'displayModeBar': False})