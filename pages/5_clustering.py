import streamlit as st
import pandas as pd
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform
from database.query import (
    get_preprocessing,
    save_hasil_clustering,
    delete_preprocessing,
    delete_hasil_clustering
)    
from models.hdbscan_model import run_hdbscan

st.set_page_config(page_title="Clustering HDBSCAN", layout="wide")
st.title("CLUSTERING HDBSCAN", anchor=False)
st.caption("Proses & Tahapan Pemodelan Clustering Menggunakan Algoritma HDBSCAN")
st.markdown("---")

# =============================================================================
# 1. AMBIL & PROTEKSI DATA PREPROCESSING
# =============================================================================
data = get_preprocessing()

if not data or len(data) == 0:
    st.warning("⚠️ Data preprocessing masih kosong. Silakan lakukan proses preprocessing terlebih dahulu.")
    delete_hasil_clustering()
    st.stop()

df = pd.DataFrame(data)
df.columns = [col.upper() for col in df.columns]

# Deteksi Fitur Seleksi Secara Dinamis (Mengabaikan kolom identitas)
identitas_cols = ['id', 'TAHUN', 'KECAMATAN', 'created_at', 'updated_at']
fitur_cols = [col for col in df.columns if col.lower() not in [c.lower() for c in identitas_cols]]

# Matriks fitur terstandarisasi (Z-Score)
X_scaled = df[fitur_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values

# Label Objek Data (Kecamatan)
labels_obj = [
    f"{i+1}. {r['KECAMATAN']} ({r['TAHUN']})" if 'TAHUN' in df.columns else f"{i+1}. {r['KECAMATAN']}" 
    for i, (_, r) in enumerate(df.iterrows())
]

# =============================================================================
# 2. TAMPILAN DEFAULT: DATA HASIL STANDARDISASI (INPUT HDBSCAN)
# =============================================================================
st.subheader("1. Data Hasil Standardisasi (Input HDBSCAN)", anchor=False)
st.caption(f"Matriks data produktivitas {len(fitur_cols)} komoditas per kecamatan yang telah melalui tahapan **Standardisasi (Z-Score)** pada halaman Preprocessing.")

col_info1, col_info2 = st.columns([2, 1])
with col_info1:
    # Tabel Data Hasil Standardisasi
    display_cols = ['KECAMATAN'] + fitur_cols if 'KECAMATAN' in df.columns else fitur_cols
    st.dataframe(df[display_cols], width="stretch", hide_index=True)

with col_info2:
    st.info(f"""
    **Ringkasan Input:**
    * **Jumlah Kecamatan (N):** {len(df)} objek data
    * **Jumlah Komoditas (D):** {len(fitur_cols)} fitur
    * **Daftar Komoditas:** {', '.join(fitur_cols)}
    """)

# Contoh Vektor Dua Kecamatan
if len(df) >= 2 and 'KECAMATAN' in df.columns:
    kec_1_name = df.iloc[0]['KECAMATAN']
    kec_2_name = df.iloc[1]['KECAMATAN']
    vec_1 = X_scaled[0]
    vec_2 = X_scaled[1]

    with st.expander(f"🔍 Lihat Contoh Vektor Terstandarisasi ({kec_1_name} vs {kec_2_name})"):
        st.write(f"**{kec_1_name} ($X_1$):** `{np.round(vec_1, 4).tolist()}`")
        st.write(f"**{kec_2_name} ($X_2$):** `{np.round(vec_2, 4).tolist()}`")
        st.caption("Nilai vektor mewakili pola produktivitas relatif komoditas yang telah di-scale (rata-rata = 0, varians = 1).")

st.markdown("---")

# =============================================================================
# 3. PENGATURAN PARAMETER HDBSCAN
# =============================================================================
st.subheader("2. Parameter Algoritma HDBSCAN", anchor=False)
st.caption("💡 **Tips Parameter:** Untuk dataset tingkat kecamatan (±31 data), gunakan `min_cluster_size = 2` atau `3` agar kelompok data tidak terdeteksi sebagai Noise seluruhnya.")

col_p1, col_p2 = st.columns(2)
with col_p1:
    min_cluster_size = st.number_input(
        "Minimum Cluster Size (min_cluster_size)",
        min_value=2,
        max_value=len(df),
        value=2,
        help="Jumlah minimal objek data untuk dapat membentuk sebuah klaster sah.",
        key="input_min_cluster_size"
    )
with col_p2:
    min_samples = st.number_input(
        "Minimum Samples (min_samples)",
        min_value=1,
        max_value=len(df),
        value=1,
        help="Jumlah tetangga terdekat (k) untuk menghitung k-NN distance / Core Distance.",
        key="input_min_samples"
    )

st.markdown("---")

# =============================================================================
# 4. TAHAPAN MATEMATIKA HDBSCAN
# =============================================================================
st.subheader("3. Tahapan Matematika & Perhitungan HDBSCAN", anchor=False)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Euclidean Distance",
    "2. Core Distance",
    "3. Mutual Reachability Distance (MRD)",
    "4. Minimum Spanning Tree (MST)",
    "5. Cluster Stability"
])

# -----------------------------------------------------------------------------
# TAB 1: EUCLIDEAN DISTANCE
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("#### TAHAP 1: EUCLIDEAN DISTANCE")
    st.write("Sistem menghitung matriks jarak Euclidean antar seluruh pasangan kecamatan berdasarkan pola produktivitas komoditas yang telah di-standardisasi.")
    
    st.latex(r"d(A, B) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}")
    
    st.markdown(rf"""
    **Keterangan:**
    * $d(A, B)$ : Jarak Euclidean antara Kecamatan $A$ dan Kecamatan $B$.
    * $n$ : Jumlah fitur komoditas (**{len(fitur_cols)} komoditas**).
    * $A_i, B_i$ : Nilai produktivitas terstandarisasi komoditas ke-$i$.
    """)
    
    # Perhitungan Matriks Euclidean Distance
    dist_matrix = squareform(pdist(X_scaled, metric='euclidean'))
    df_dist = pd.DataFrame(dist_matrix, index=labels_obj, columns=labels_obj)
    
    st.write(f"##### Matriks Jarak Euclidean ({len(df)} × {len(df)} Kecamatan):")
    st.dataframe(df_dist, width="stretch")

# -----------------------------------------------------------------------------
# TAB 2: CORE DISTANCE
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("#### TAHAP 2: CORE DISTANCE")
    st.write("Core Distance mengukur kepadatan lokal dengan mencari jarak dari suatu kecamatan ke tetangga terdekat ke-$k$.")
    
    st.latex(r"\text{Core}_k(x) = d(x, N_k(x))")
    
    st.markdown(rf"""
    **Keterangan:**
    * $x$ : Objek Kecamatan.
    * $N_k(x)$ : Tetangga terdekat ke-$k$ dari kecamatan $x$ (dengan $k = \text{{min\_samples}} = {min_samples}$).
    """)
    
    # Perhitungan Core Distance
    sorted_dists = np.sort(dist_matrix, axis=1)
    k_index = min(int(min_samples) - 1, sorted_dists.shape[1] - 1)
    core_distances = sorted_dists[:, k_index]

    df_core = pd.DataFrame({
        'NO': range(1, len(df) + 1),
        'KECAMATAN': df['KECAMATAN'] if 'KECAMATAN' in df.columns else labels_obj,
        'Core Distance': core_distances
    })
    
    st.write(f"##### Tabel Nilai Core Distance ($k = {min_samples}$):")
    st.dataframe(df_core, width="stretch", hide_index=True)

# -----------------------------------------------------------------------------
# TAB 3: MUTUAL REACHABILITY DISTANCE (MRD)
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("#### TAHAP 3: MUTUAL REACHABILITY DISTANCE (MRD)")
    st.write("MRD menyesuaikan jarak antar kecamatan berdasarkan kepadatan lokalnya. Daerah padat akan mempertahankan jarak aslinya, sedangkan daerah renggang diperlebar jaraknya.")
    
    st.latex(r"d_{\text{mreach}}(a, b) = \max \left( \text{Core}_k(a), \, \text{Core}_k(b), \, d(a, b) \right)")
    
    # Perhitungan Matriks MRD
    mrd_matrix = np.maximum(
        np.maximum(core_distances[:, None], core_distances[None, :]),
        dist_matrix
    )

    df_mrd = pd.DataFrame(mrd_matrix, index=labels_obj, columns=labels_obj)
    
    st.write("##### Matriks Mutual Reachability Distance (MRD):")
    st.dataframe(df_mrd, width="stretch")

# -----------------------------------------------------------------------------
# TAB 4: MINIMUM SPANNING TREE (MST)
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("#### TAHAP 4: MINIMUM SPANNING TREE (MST)")
    st.write("Berdasarkan matriks MRD, HDBSCAN membangun graf terhubung (*Minimum Spanning Tree*) yang menghubungkan seluruh kecamatan dengan total bobot jarak terkecil tanpa membentuk siklus.")
    
    st.latex(r"G = (V, E)")
    
    st.markdown("""
    **Prinsip Kerja:**
    1. Seluruh kecamatan diawali sebagai $N$ kelompok terpisah.
    2. Sisi graf (*edge*) dengan jarak MRD terbesar dipotong secara bertahap.
    3. Pemotongan sisi menghasilkan pembagian hierarki klaster (*Condensed Dendrogram Tree*).
    """)
    st.info("💡 **Integrasi Modul:** Konstruksi graf MST dan hirarki kondensasi diproses secara penuh pada fungsi `models/hdbscan_model.py` saat proses clustering dijalankan.")

# -----------------------------------------------------------------------------
# TAB 5: CLUSTER STABILITY
# -----------------------------------------------------------------------------
with tab5:
    st.markdown("#### TAHAP 5: CLUSTER STABILITY")
    st.write("HDBSCAN mengevaluasi tingkat ketahanan (*persistence*) tiap klaster pada hirarki kepadatan $\lambda$ untuk menentukan klaster akhir yang paling stabil.")
    
    st.latex(r"\lambda = \frac{1}{\text{distance}}")
    st.latex(r"\text{Stability}(C) = \sum_{p \in C} \left( \lambda_{\text{death}}(p, C) - \lambda_{\text{birth}}(C) \right)")
    
    st.success("✅ **Seleksi Otomatis:** Klaster dengan nilai stability tertinggi dipertahankan, sedangkan titik data yang tidak stabil dialokasikan sebagai **Noise (-1)**.")

st.markdown("---")

# =============================================================================
# 5. TOMBOL EKSEKUSI & RESET
# =============================================================================
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("Jalankan Clustering", width="stretch", type="primary"):
        current_data = get_preprocessing()
        if not current_data or len(current_data) == 0:
            st.error("Gagal menjalankan: Dataset mendadak kosong/terhapus!")
            st.rerun()
            
        with st.spinner("Sedang memproses algoritma HDBSCAN..."):
            # Simpan parameter ke session_state untuk dibaca di halaman 6_hasil_cluster.py
            st.session_state['param_min_cluster_size'] = min_cluster_size
            st.session_state['param_min_samples'] = min_samples

            hasil = run_hdbscan(df, min_cluster_size, min_samples)
            berhasil, error_msg = save_hasil_clustering(hasil)

            if berhasil:
                st.success("Clustering berhasil dilakukan dan data berhasil disimpan!")
                time.sleep(1.2)
                st.switch_page("pages/6_hasil_cluster.py")
            else:
                st.error(f"❌ Gagal menyimpan ke database. Detail Error:\n\n`{error_msg}`")

with col_btn2:
    if st.button("Reset Data Preprocessing", width="stretch"):
        if delete_preprocessing():
            delete_hasil_clustering()
            st.success("Data Preprocessing dan hasil clustering berhasil dibersihkan!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Gagal membersihkan data.")