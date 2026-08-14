import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from hdbscan.validity import validity_index

from database.query import get_hasil_clustering

# -----------------------------------------------------------------------------
# 1. FUNGSI PERHITUNGAN METRIK EVALUASI (DBCV & SILHOUETTE)
# -----------------------------------------------------------------------------
def hitung_evaluasi_hdbscan(X, labels):
    """
    Menghitung evaluasi DBCV Score dan Silhouette Score untuk HDBSCAN.
    - X: data fitur terstandarisasi (numpy array float64)
    - labels: array label klaster hasil HDBSCAN (termasuk label -1)
    """
    # HITUNG DBCV SCORE
    try:
        X_dbcv = np.ascontiguousarray(X, dtype=np.float64)
        labels_dbcv = np.ascontiguousarray(labels, dtype=np.int64)
        
        # Minimal ada 2 kelompok label berbeda untuk DBCV
        if len(np.unique(labels_dbcv)) > 1:
            dbcv_val = validity_index(X_dbcv, labels_dbcv)
            dbcv_str = f"{dbcv_val:.4f}".replace(".", ",")
        else:
            dbcv_str = "N/A"
    except Exception:
        dbcv_str = "N/A"

    # HITUNG SILHOUETTE SCORE (Hanya data non-noise)
    non_noise_mask = labels != -1
    unique_clusters = set(labels[non_noise_mask])
    
    if len(unique_clusters) > 1 and np.sum(non_noise_mask) > len(unique_clusters):
        try:
            sil_val = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
            sil_str = f"{sil_val:.4f}".replace(".", ",")
        except Exception:
            sil_str = "N/A"
    else:
        sil_str = "N/A"
        
    return dbcv_str, sil_str


# -----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Hasil Clustering HDBSCAN", layout="wide")
st.title("HASIL CLUSTERING", anchor=False)
st.caption("Hasil Clustering Menggunakan Algoritma HDBSCAN")

# -----------------------------------------------------------------------------
# LOAD & PREPROCESSING DATA
# -----------------------------------------------------------------------------
# LOAD & PREPROCESSING DATA
# -----------------------------------------------------------------------------
data_hasil = get_hasil_clustering()
if not data_hasil or len(data_hasil) == 0:
    st.warning("⚠️ Belum ada hasil clustering. Silakan jalankan proses clustering HDBSCAN terlebih dahulu pada halaman 5_clustering.")
    st.stop()

df = pd.DataFrame(data_hasil)
df.columns = [str(col).upper().strip() for col in df.columns]

# Deteksi Kolom Fitur Komoditas (Z-Score) Secara Dinamis
non_fitur_cols = ['ID', 'ID_PREPROCESSING', 'TAHUN', 'KECAMATAN', 'CLUSTER', 'PROBABILITY', 'OUTLIER_SCORE', 'CLUSTER_FINAL', 'KATEGORI', 'CREATED_AT', 'UPDATED_AT']
fitur_cols = [col for col in df.columns if col.upper() not in non_fitur_cols]

# Pastikan nilai numerik pada fitur komoditas valid
for col in fitur_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

# Filter Tahun Dataset jika terdapat lebih dari 1 tahun di hasil clustering
if "TAHUN" in df.columns and df["TAHUN"].dropna().any():
    daftar_tahun_hasil = sorted(df["TAHUN"].dropna().unique().tolist(), reverse=True)
    if len(daftar_tahun_hasil) > 1:
        col_t1, _ = st.columns([1, 3])
        with col_t1:
            opsi_t = ["Semua Tahun"] + [str(int(t)) for t in daftar_tahun_hasil]
            def_idx = 0
            if "selected_tahun" in st.session_state and st.session_state["selected_tahun"] is not None:
                t_str = str(int(st.session_state["selected_tahun"]))
                if t_str in opsi_t:
                    def_idx = opsi_t.index(t_str)
            t_pilih = st.selectbox("📅 Filter Tahun Hasil Clustering:", options=opsi_t, index=def_idx, key="filter_tahun_hasil")
            
        if t_pilih != "Semua Tahun":
            df = df[df["TAHUN"] == int(t_pilih)].copy()

# -----------------------------------------------------------------------------
# LOGIKA PEMETAAN CLUSTER (0: Tinggi, 1: Sedang, 2: Rendah, -1: Noise)
# -----------------------------------------------------------------------------
cluster_col = 'CLUSTER' if 'CLUSTER' in df.columns else 'CLUSTER_FINAL'
if cluster_col not in df.columns:
    st.error("❌ Kolom 'CLUSTER' tidak ditemukan dalam tabel database.")
    st.stop()

df_valid = df[df[cluster_col] != -1].copy()

if len(df_valid) > 0:
    # Urutkan cluster berdasarkan rata-rata komoditas tertinggi ke terendah
    if "PRODUKTIVITAS" in df_valid.columns:
        cluster_means = df_valid.groupby(cluster_col)["PRODUKTIVITAS"].mean().sort_values(ascending=False)
    else:
        cluster_means = df_valid.groupby(cluster_col)[fitur_cols].mean().mean(axis=1).sort_values(ascending=False)
        
    n_klaster_found = len(cluster_means)
    
    orig_to_new = {}
    orig_to_kategori = {}
    
    for idx, orig_id in enumerate(cluster_means.index):
        if n_klaster_found >= 3:
            if idx == 0:
                new_id = 0
                kat = "Produktivitas Tinggi"
            elif idx == n_klaster_found - 1:
                new_id = 2
                kat = "Produktivitas Rendah"
            else:
                new_id = 1
                kat = "Produktivitas Sedang"
        elif n_klaster_found == 2:
            if idx == 0:
                new_id = 0
                kat = "Produktivitas Tinggi"
            else:
                new_id = 2
                kat = "Produktivitas Rendah"
        else:
            new_id = 1
            kat = "Produktivitas Sedang"
            
        orig_to_new[orig_id] = new_id
        orig_to_kategori[orig_id] = kat

    orig_to_new[-1] = -1
    orig_to_kategori[-1] = "Noise (Outlier)"

    df["CLUSTER_FINAL"] = df[cluster_col].map(orig_to_new).fillna(-1).astype(int)
    df["KATEGORI"] = df[cluster_col].map(orig_to_kategori).fillna("Noise (Outlier)")
else:
    df["CLUSTER_FINAL"] = -1
    df["KATEGORI"] = "Noise (Outlier)"

# -----------------------------------------------------------------------------
# PERHITUNGAN METRIK EVALUASI
# -----------------------------------------------------------------------------
# X_scaled langsung diambil dari fitur komoditas yang sudah terstandarisasi Z-Score
X_scaled = df[fitur_cols].values
labels = df['CLUSTER_FINAL'].values

# Panggil fungsi evaluasi terpusat
dbcv_str, sil_str = hitung_evaluasi_hdbscan(X_scaled, labels)

# Integrasi parameter dinamis dari session state
min_cluster_size = st.session_state.get("param_min_cluster_size", 2)
min_samples = st.session_state.get("param_min_samples", 1)

total_data = len(df)
unique_clusters = set(labels) - {-1}
n_cluster_count = len(unique_clusters)
n_noise = (df["CLUSTER_FINAL"] == -1).sum()

# Tampilkan ringkasan
ringkasan_box = f"""==========================================
        HASIL CLUSTERING HDBSCAN
==========================================

Jumlah Data             : {total_data}
Jumlah Cluster          : {n_cluster_count}
Jumlah Noise            : {n_noise}
min_cluster_size        : {min_cluster_size}
min_samples             : {min_samples}
DBCV Score              : {dbcv_str}
Silhouette Score        : {sil_str}
=========================================="""

st.code(ringkasan_box, language=None)
st.markdown("---")

# Mapping warna dan order
color_map = {
    "Produktivitas Tinggi": "#2CA02C", 
    "Produktivitas Sedang": "#FFBB11", 
    "Produktivitas Rendah": "#D62728", 
    "Noise (Outlier)": "#7F7F7F"       
}
category_orders = ["Produktivitas Tinggi", "Produktivitas Sedang", "Produktivitas Rendah", "Noise (Outlier)"]

# -----------------------------------------------------------------------------
# 2. TABEL HASIL CLUSTERING
# -----------------------------------------------------------------------------
st.subheader("Tabel Hasil Clustering", anchor=False)

col_f1, _ = st.columns([1, 3])
with col_f1:
    pilihan_kategori = st.multiselect(
        "Filter Kategori:",
        options=category_orders,
        default=[k for k in category_orders if k in df["KATEGORI"].unique()]
    )

df_filtered = df[df['KATEGORI'].isin(pilihan_kategori)].copy()

kolom_tampil = ["TAHUN", "KECAMATAN"] + fitur_cols + ["OUTLIER_SCORE", "KATEGORI", "CLUSTER_FINAL"]
kolom_tampil = [c for c in kolom_tampil if c in df_filtered.columns]

st.dataframe(
    df_filtered[kolom_tampil].rename(columns={"CLUSTER_FINAL": "CLUSTER"}),
    width="stretch",
    hide_index=True
)
st.markdown("---")

# -----------------------------------------------------------------------------
# 3. TABEL STATISTIK SETIAP CLUSTER
# -----------------------------------------------------------------------------
st.subheader("Tabel Statistik Setiap Cluster", anchor=False)

if len(df) > 0:
    stat_df = df.groupby(["CLUSTER_FINAL", "KATEGORI"]).agg(
        JUMLAH_DATA=("KECAMATAN", "count") if "KECAMATAN" in df.columns else (fitur_cols[0], "count"),
        RATA_OUTLIER_SCORE=("OUTLIER_SCORE", "mean") if "OUTLIER_SCORE" in df.columns else (fitur_cols[0], "count")
    ).reset_index()

    # Hitung rata-rata tiap fitur komoditas per klaster
    mean_komoditas = df.groupby(["CLUSTER_FINAL", "KATEGORI"])[fitur_cols].mean().reset_index()
    stat_df = pd.merge(stat_df, mean_komoditas, on=["CLUSTER_FINAL", "KATEGORI"])

    stat_df = stat_df.rename(
        columns={
            "CLUSTER_FINAL": "Cluster",
            "KATEGORI": "Kategori",
            "JUMLAH_DATA": "Jumlah Data",
            "RATA_OUTLIER_SCORE": "Rata-Rata Outlier Score"
        }
    )

    if "Rata-Rata Outlier Score" in stat_df.columns:
        stat_df["Rata-Rata Outlier Score"] = stat_df["Rata-Rata Outlier Score"].map("{:.4f}".format)

    st.dataframe(stat_df, width="stretch", hide_index=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 4. GRAFIK PLOT HDBSCAN CLUSTERING
# -----------------------------------------------------------------------------
st.subheader("Grafik Plot Clustering HDBSCAN", anchor=False)
st.caption("Visualisasi sebaran klaster produktivitas hasil pertanian.")

df_plot = df.copy()

cluster_config = {
    0: {"color": "#2ca02c", "label": "Cluster Tinggi (Produktivitas Tinggi)"},
    1: {"color": "#1f77b4", "label": "Cluster Sedang (Produktivitas Sedang)"},
    2: {"color": "#ff7f0e", "label": "Cluster Rendah (Produktivitas Rendah)"},
    -1: {"color": "#7f7f7f", "label": "Noise / Outlier"},
}
fig, ax = plt.subplots(figsize=(9, 5.5))

# Gunakan 2 fitur komoditas pertama jika tersedia untuk koordinat 2D
plot_x_col = fitur_cols[0] if len(fitur_cols) > 0 else None
plot_y_col = fitur_cols[1] if len(fitur_cols) > 1 else (fitur_cols[0] if len(fitur_cols) > 0 else None)

if plot_x_col and plot_y_col:
    for c_id, config in cluster_config.items():
        sub_df = df_plot[df_plot["CLUSTER_FINAL"] == c_id]

        if not sub_df.empty:
            ax.scatter(
                sub_df[plot_x_col],
                sub_df[plot_y_col],
                c=config["color"],
                label=f"{config['label']} ({len(sub_df)} Data)",
                s=65,
                edgecolors="black",
                linewidth=0.7,
                alpha=0.8,
            )

    ax.set_title(
        f"Sebaran Klaster ({plot_x_col} vs {plot_y_col})",
        fontsize=12,
        pad=12,
        weight="bold",
    )
    ax.set_xlabel(f"{plot_x_col} (Z-Score)", fontsize=10)
    ax.set_ylabel(f"{plot_y_col} (Z-Score)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9, frameon=True)

    st.pyplot(fig)
else:
    st.info("Fitur untuk visualisasi scatter plot belum tersedia.")

st.markdown("---")

# -----------------------------------------------------------------------------
# 5. DAFTAR KECAMATAN BERDASARKAN KLASTER
# -----------------------------------------------------------------------------
st.subheader("Daftar Kecamatan Berdasarkan Klaster", anchor=False)
st.caption("Rincian tabel objek data (Kecamatan) yang tergolong ke dalam masing-masing kelompok klaster.")

cluster_structure = [
    (0, "Cluster Tinggi (Produktivitas Tinggi)"),
    (1, "Cluster Sedang (Produktivitas Sedang)"),
    (2, "Cluster Rendah (Produktivitas Rendah)"),
    (-1, "Noise / Outlier")
]

for c_id, c_title in cluster_structure:
    df_sub = df[df["CLUSTER_FINAL"] == c_id].copy()
    
    st.markdown(f"##### 📌 {c_title} ({len(df_sub)} Objek Data)")
    
    if not df_sub.empty:
        sort_cols = [c for c in ['KECAMATAN'] if c in df_sub.columns]
        if sort_cols:
            df_sub = df_sub.sort_values(by=sort_cols).reset_index(drop=True)
            
        df_sub.insert(0, 'NO', range(1, len(df_sub) + 1))
        
        kolom_tampil = ['NO', 'TAHUN', 'KECAMATAN'] + fitur_cols + ['OUTLIER_SCORE']
        kolom_tersedia = [c for c in kolom_tampil if c in df_sub.columns]
        
        st.dataframe(
            df_sub[kolom_tersedia],
            width="stretch",
            hide_index=True
        )
    else:
        st.info("Tidak ada objek data pada klaster ini.")
        
    st.write("")
    st.markdown("---")

# -----------------------------------------------------------------------------
# 6. DISTRIBUSI CLUSTER DIAGRAM BATANG
# -----------------------------------------------------------------------------
st.subheader("Distribusi Cluster (Diagram Batang)", anchor=False)

dist_df = df.groupby(["CLUSTER_FINAL", "KATEGORI"]).size().reset_index(name="JUMLAH_DATA")
dist_df["LABEL"] = "Cluster " + dist_df["CLUSTER_FINAL"].astype(str) + " (" + dist_df["KATEGORI"] + ")"

fig_bar = px.bar(
    dist_df,
    x="LABEL",
    y="JUMLAH_DATA",
    color="KATEGORI",
    color_discrete_map=color_map,
    text="JUMLAH_DATA",
    title="Jumlah Data Pada Masing-Masing Cluster",
    labels={"LABEL": "Cluster", "JUMLAH_DATA": "Jumlah Data"},
    template="plotly_white"
)

fig_bar.update_traces(textposition='outside')
fig_bar.update_layout(height=420, showlegend=False)

st.plotly_chart(fig_bar, width="stretch")

st.markdown("---")

# -----------------------------------------------------------------------------
# 7. VISUALISASI PER KECAMATAN
# -----------------------------------------------------------------------------
st.subheader("Detail Per Kecamatan", anchor=False)

if "KECAMATAN" in df.columns and df["KECAMATAN"].dropna().any():
    daftar_kecamatan = sorted(df["KECAMATAN"].dropna().unique())
    kec_terpilih = st.selectbox("Pilih Kecamatan untuk Melihat Detail Nilai Fitur & Klaster:", options=daftar_kecamatan)

    df_kec = df[df["KECAMATAN"] == kec_terpilih].copy()

    st.markdown(f"##### Informasi Pertanian Kecamatan: **{kec_terpilih}**")

    col_k1, col_k2 = st.columns(2)
    kategori_kec = df_kec['KATEGORI'].iloc[0] if 'KATEGORI' in df_kec.columns else '-'
    col_k1.metric("Status Klaster", f"{kategori_kec}")
    outlier_kec = df_kec['OUTLIER_SCORE'].iloc[0] if 'OUTLIER_SCORE' in df_kec.columns else 0.0
    col_k2.metric("Outlier Score", f"{float(outlier_kec):.4f}")

    kolom_kec_tampil = ["TAHUN", "KECAMATAN"] + fitur_cols + ["OUTLIER_SCORE", "KATEGORI", "CLUSTER_FINAL"]
    kolom_kec_tampil = [c for c in kolom_kec_tampil if c in df_kec.columns]

    st.dataframe(
        df_kec[kolom_kec_tampil].rename(columns={"CLUSTER_FINAL": "CLUSTER"}),
        width="stretch",
        hide_index=True
    )
else:
    st.info("Kolom KECAMATAN tidak tersedia pada dataset ini.")