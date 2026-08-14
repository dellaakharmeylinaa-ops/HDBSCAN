import pandas as pd
import plotly.express as px
import streamlit as st

from database.query import get_hasil_clustering

# Konfigurasi Halaman
st.set_page_config(page_title="Visualisasi Klaster Pertanian", layout="wide")

st.title("Visualisasi Klaster Pertanian", anchor=False)
st.caption(
    "Eksplorasi dan Distribusi Data Produktivitas Hasil Pertanian Kab."
    " Indramayu"
)
st.markdown("---")

# 1. Ambil data hasil clustering dari database
data_hasil = get_hasil_clustering()

if not data_hasil or len(data_hasil) == 0:
    st.warning(
        "⚠️ Belum ada hasil clustering. Silakan jalankan algoritma HDBSCAN di"
        " menu Clustering terlebih dahulu."
    )
    st.stop()

df = pd.DataFrame(data_hasil)
df.columns = [str(col).upper().strip() for col in df.columns]

# Deteksi Kolom Fitur Komoditas (Z-Score) Secara Dinamis
non_fitur_cols = ['ID', 'ID_PREPROCESSING', 'TAHUN', 'KECAMATAN', 'CLUSTER', 'PROBABILITY', 'OUTLIER_SCORE', 'CLUSTER_FINAL', 'KATEGORI', 'CREATED_AT', 'UPDATED_AT']
fitur_cols = [c for c in df.columns if c.upper() not in non_fitur_cols]

# Pastikan nilai numerik pada fitur komoditas valid
for col in fitur_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

# 2. Logika pemetaan klaster (Konsisten: 3 Klaster + Noise)
df_valid = df[df["CLUSTER"] != -1].copy()

if len(df_valid) > 0:
    # Urutkan klaster berdasarkan rata-rata produktivitas secara descending
    if "PRODUKTIVITAS" in df_valid.columns:
        cluster_means = df_valid.groupby("CLUSTER")["PRODUKTIVITAS"].mean().sort_values(ascending=False)
    elif fitur_cols:
        cluster_means = df_valid.groupby("CLUSTER")[fitur_cols].mean().mean(axis=1).sort_values(ascending=False)
    else:
        cluster_means = df_valid.groupby("CLUSTER").size().sort_values(ascending=False)

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

    df["CLUSTER_FINAL"] = (
        df["CLUSTER"].map(orig_to_new).fillna(-1).astype(int)
    )
    df["KATEGORI"] = (
        df["CLUSTER"].map(orig_to_kategori).fillna("Noise (Outlier)")
    )
else:
    df["CLUSTER_FINAL"] = -1
    df["KATEGORI"] = "Noise (Outlier)"

# Format Label Gabungan
df["LABEL_KLASTER"] = (
    "Cluster "
    + df["CLUSTER_FINAL"].astype(str)
    + " ("
    + df["KATEGORI"]
    + ")"
)

# Pemetaan Warna Konsisten
color_map = {
    "Produktivitas Tinggi": "#2CA02C",  # Hijau
    "Produktivitas Sedang": "#FFBB11",  # Kuning
    "Produktivitas Rendah": "#D62728",  # Merah
    "Noise (Outlier)": "#7F7F7F",  # Abu-abu
}

category_orders = [
    "Produktivitas Tinggi",
    "Produktivitas Sedang",
    "Produktivitas Rendah",
    "Noise (Outlier)",
]

# -------------------------------------------------------------
# VISUALISASI 1: Distribusi Anggota Klaster (Pie & Bar Chart)
# -------------------------------------------------------------
st.subheader("Proporsi Data per Klaster", anchor=False)
st.caption(
    "Menampilkan persentase sebaran data dan jumlah objek data pada setiap"
    " kelompok klaster."
)

col_v1, col_v2 = st.columns(2)

cluster_counts = (
    df.groupby(["CLUSTER_FINAL", "KATEGORI"])
    .size()
    .reset_index(name="JUMLAH_DATA")
)

with col_v1:
    # Pie Chart
    fig_pie = px.pie(
        cluster_counts,
        names="KATEGORI",
        values="JUMLAH_DATA",
        title="Persentase Sebaran Klaster",
        hole=0.4,  # Shape donut chart
        color="KATEGORI",
        color_discrete_map=color_map,
        category_orders={"KATEGORI": category_orders},
    )
    fig_pie.update_traces(
        textinfo="percent+label",
        hovertemplate="%{label}: %{value} Data (%{percent})",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_v2:
    # Bar Chart
    fig_bar = px.bar(
        cluster_counts,
        x="KATEGORI",
        y="JUMLAH_DATA",
        text="JUMLAH_DATA",
        title="Jumlah Objek Data per Klaster",
        color="KATEGORI",
        color_discrete_map=color_map,
        category_orders={"KATEGORI": category_orders},
        labels={"KATEGORI": "Kelompok Klaster", "JUMLAH_DATA": "Jumlah Data"},
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        showlegend=False, xaxis_title="", yaxis_title="Jumlah Data"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# -------------------------------------------------------------
# VISUALISASI 2: Boxplot (Analisis Variansi Antar Klaster)
# -------------------------------------------------------------
st.subheader("Analisis Rentang Nilai (Boxplot)", anchor=False)
st.write(
    "Boxplot digunakan untuk melihat sebaran data, nilai median, dan pencilan"
    " (outlier) dari setiap klaster terhadap variabel komoditas pertanian."
)

if fitur_cols:
    # Pilihan variabel untuk boxplot
    opsi_var = st.selectbox(
        "Pilih Variabel Komoditas untuk dianalisis:",
        fitur_cols,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    hover_cols = [col for col in ["KECAMATAN", "TAHUN", "OUTLIER_SCORE"] if col in df.columns]

    fig_box = px.box(
        df,
        x="KATEGORI",
        y=opsi_var,
        color="KATEGORI",
        color_discrete_map=color_map,
        category_orders={"KATEGORI": category_orders},
        points="all",  # Menampilkan titik data di samping boxplot
        title=f"Sebaran Nilai {opsi_var.replace('_', ' ').title()} (Z-Score) pada Setiap Klaster",
        labels={
            "KATEGORI": "Kelompok Klaster",
            opsi_var: f"{opsi_var.replace('_', ' ').title()} (Z-Score)",
        },
        hover_data=hover_cols,
    )
    fig_box.update_layout(showlegend=False, xaxis_title="Kelompok Klaster")
    st.plotly_chart(fig_box, use_container_width=True)
else:
    st.info("Fitur komoditas belum tersedia untuk boxplot.")

st.markdown("---")

# -------------------------------------------------------------
# VISUALISASI 3: 3D Scatter Plot
# -------------------------------------------------------------
st.subheader("Ruang Fitur 3 Dimensi", anchor=False)
st.write(
    "Melihat posisi kecamatan dalam ruang 3 dimensi berdasarkan fitur komoditas yang dipilih."
)

if len(fitur_cols) >= 3:
    c_x, c_y, c_z = st.columns(3)
    with c_x:
        x_dim = st.selectbox("Sumbu X:", fitur_cols, index=0, format_func=lambda x: x.replace("_", " ").title(), key="dim_x")
    with c_y:
        y_dim = st.selectbox("Sumbu Y:", fitur_cols, index=1 if len(fitur_cols) > 1 else 0, format_func=lambda x: x.replace("_", " ").title(), key="dim_y")
    with c_z:
        z_dim = st.selectbox("Sumbu Z:", fitur_cols, index=2 if len(fitur_cols) > 2 else 0, format_func=lambda x: x.replace("_", " ").title(), key="dim_z")

    hover_cols = [c for c in ["KECAMATAN", "TAHUN", "OUTLIER_SCORE"] if c in df.columns]

    fig_3d = px.scatter_3d(
        df,
        x=x_dim,
        y=y_dim,
        z=z_dim,
        color="KATEGORI",
        color_discrete_map=color_map,
        category_orders={"KATEGORI": category_orders},
        hover_name="KECAMATAN" if "KECAMATAN" in df.columns else None,
        hover_data=hover_cols,
        title=f"Pemetaan 3D Klaster ({x_dim.replace('_', ' ').title()} vs {y_dim.replace('_', ' ').title()} vs {z_dim.replace('_', ' ').title()})",
        opacity=0.85,
        labels={
            x_dim: f"{x_dim.replace('_', ' ').title()} (Z-Score)",
            y_dim: f"{y_dim.replace('_', ' ').title()} (Z-Score)",
            z_dim: f"{z_dim.replace('_', ' ').title()} (Z-Score)",
            "KATEGORI": "Klaster",
        },
    )

    fig_3d.update_traces(
        marker=dict(size=6, line=dict(width=0.5, color="DarkSlateGrey"))
    )
    fig_3d.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1),
    )
    st.plotly_chart(fig_3d, use_container_width=True)
elif len(fitur_cols) >= 2:
    fig_2d = px.scatter(
        df,
        x=fitur_cols[0],
        y=fitur_cols[1],
        color="KATEGORI",
        color_discrete_map=color_map,
        category_orders={"KATEGORI": category_orders},
        hover_name="KECAMATAN" if "KECAMATAN" in df.columns else None,
        title=f"Pemetaan 2D Klaster ({fitur_cols[0]} vs {fitur_cols[1]})"
    )
    st.plotly_chart(fig_2d, use_container_width=True)
else:
    st.info("Fitur belum mencukupi untuk visualisasi 3 Dimensi.")