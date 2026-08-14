import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from database.query import get_hasil_clustering

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
except Exception as e:
    st.error(f"Gagal memuat data dari database. Error: {e}")
    st.stop()

if not data or len(data) == 0:
    st.warning("Belum ada data clustering yang tersedia. Silahkan admin untuk melakukan proses clustering terlebih dahulu.")
    st.stop()

df = pd.DataFrame(data)

# STANDARDISASI NAMA KOLOM (Huruf Kapital & Tanpa Spasi)
df.columns = [str(col).upper().strip() for col in df.columns]

# VALIDASI WAJIB: Cek apakah kolom CLUSTER ada di DataFrame
if "CLUSTER" not in df.columns:
    st.warning("⚠️ Kolom 'CLUSTER' belum ditemukan pada dataset. Pastikan Anda sudah menjalankan pemodelan di halaman **Clustering HDBSCAN**.")
    st.stop()

# Konversi kolom CLUSTER ke numerik secara aman
df["CLUSTER"] = pd.to_numeric(df["CLUSTER"], errors="coerce").fillna(-1).astype(int)

# -----------------------------------------------------------------------------
# 1. LOGIKA PEMETAAN KATEGORI CLUSTER (DENGAN PENANGANAN DATA PIVOT / LONG)

# Cek apakah kolom PRODUKTIVITAS tunggal ada atau data berbentuk Pivot (banyak kolom)
if "PRODUKTIVITAS" in df.columns:
    df["PRODUKTIVITAS"] = pd.to_numeric(df["PRODUKTIVITAS"], errors="coerce").fillna(0)
    cluster_mean = (
        df[df["CLUSTER"] != -1]
        .groupby("CLUSTER")["PRODUKTIVITAS"]
        .mean()
        .sort_values(ascending=False)
    )
else:
    # Jika data berbentuk pivot, ambil rata-rata dari kolom-kolom komoditas
    kolom_non_fitur = ['ID', 'ID_PREPROCESSING', 'TAHUN', 'KECAMATAN', 'CLUSTER', 'PROBABILITY', 'OUTLIER_SCORE']
    fitur_cols = [c for c in df.columns if c.upper() not in kolom_non_fitur]
    
    # Hitung rerata per cluster
    if fitur_cols:
        df[fitur_cols] = df[fitur_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        cluster_mean = (
            df[df["CLUSTER"] != -1]
            .groupby("CLUSTER")[fitur_cols]
            .mean()
            .mean(axis=1)
            .sort_values(ascending=False)
        )

# Buat pemetaan label (Cluster 0 Tinggi, Cluster 1 Sedang, dst)
cluster_map = {}
if len(cluster_mean) >= 3:
    cluster_map[cluster_mean.index[0]] = "Cluster 0 (Tinggi)"
    cluster_map[cluster_mean.index[1]] = "Cluster 1 (Sedang)"
    cluster_map[cluster_mean.index[2]] = "Cluster 2 (Rendah)"
else:
    for idx, c_id in enumerate(cluster_mean.index):
        cluster_map[c_id] = f"Cluster {idx}"

# PERBAIKAN ERROR TYPEERROR:
# Gunakan .apply() lambda agar kolom KATEGORI otomatis menjadi tipe String/Text
df["KATEGORI"] = df["CLUSTER"].apply(
    lambda x: "Noise (-1)" if x == -1 else cluster_map.get(x, f"Cluster {int(x)}")
)

color_discrete_map = {
    "Cluster 0 (Tinggi)": "#2CA02C",  # Hijau
    "Cluster 1 (Sedang)": "#1F77B4",  # Biru
    "Cluster 2 (Rendah)": "#FFBB11",  # Kuning/Oren
    "Noise (-1)": "#D62728"           # Merah
}
category_orders = ["Cluster 0 (Tinggi)", "Cluster 1 (Sedang)", "Cluster 2 (Rendah)", "Noise (-1)"]

# -----------------------------------------------------------------------------
# 2. METRIK RINGKASAN (KPI METRICS)
# -----------------------------------------------------------------------------
m1, m2, m3 = st.columns(3)
total_kecamatan = df["KECAMATAN"].nunique() if "KECAMATAN" in df.columns else 0

if "KOMODITAS" in df.columns:
    total_komoditas = df["KOMODITAS"].nunique()
    avg_prod = df["PRODUKTIVITAS"].mean()
else:
    total_komoditas = len(fitur_cols)
    avg_prod = df[fitur_cols].values.mean()

m1.metric("Total Kecamatan", f"{total_kecamatan} Kecamatan")
m2.metric("Total Komoditas Terdaftar", f"{total_komoditas} Komoditas")
m3.metric("Rata-rata Produktivitas", f"{avg_prod:.2f} Ku/Ha")

st.markdown("---")

# -----------------------------------------------------------------------------
# 3. GRAFIK & PETA SEBARAN CLUSTER
# -----------------------------------------------------------------------------
col_graph1, col_graph2 = st.columns([1, 1])

# KIRI: Grafik Produktivitas
with col_graph1:
    st.markdown("### Rata-rata Produktivitas")
    
    if "KOMODITAS" in df.columns and "PRODUKTIVITAS" in df.columns:
        df_komoditas = df.groupby("KOMODITAS")["PRODUKTIVITAS"].mean().reset_index()
        fig_bar = px.bar(
            df_komoditas, x="KOMODITAS", y="PRODUKTIVITAS", text="PRODUKTIVITAS",
            color_discrete_sequence=["#1F77B4"] 
        )
        fig_bar.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_bar.update_yaxes(title_text="Produktivitas (Ku/Ha)")
        fig_bar.update_xaxes(title_text="")
    else:
        # Jika data Pivot (Fitur Komoditas berupa Kolom)
        df_komoditas_pivot = df[fitur_cols].mean().reset_index()
        df_komoditas_pivot.columns = ["KOMODITAS", "PRODUKTIVITAS"]
        fig_bar = px.bar(
            df_komoditas_pivot, x="KOMODITAS", y="PRODUKTIVITAS", text="PRODUKTIVITAS",
            color_discrete_sequence=["#1F77B4"]
        )
        fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_bar.update_yaxes(title_text="Rata-rata Nilai/Z-Score")
        fig_bar.update_xaxes(title_text="")

    fig_bar.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_bar, width="stretch", config={'displayModeBar': False})

# KANAN: Peta / Scatter Sebaran Cluster
with col_graph2:
    st.markdown("### Sebaran Cluster Kecamatan")
    if "LATITUDE" in df.columns and "LONGITUDE" in df.columns:
        fig_map = px.scatter_mapbox(
            df, lat="LATITUDE", lon="LONGITUDE", color="KATEGORI",
            color_discrete_map=color_discrete_map, hover_name="KECAMATAN",
            zoom=9, height=400, category_orders={"KATEGORI": category_orders}
        )
        fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        # Jika tidak ada koordinat LATITUDE/LONGITUDE di database
        fig_scatter_spatial = px.scatter(
            df, x="KECAMATAN", y="CLUSTER", color="KATEGORI",
            color_discrete_map=color_discrete_map,
            category_orders={"KATEGORI": category_orders},
            title="Sebaran Cluster per Kecamatan"
        )
        fig_scatter_spatial.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_scatter_spatial, width="stretch", config={'displayModeBar': False})