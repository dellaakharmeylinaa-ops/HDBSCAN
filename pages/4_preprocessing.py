import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from database.query import get_dataset, save_preprocessing, delete_preprocessing

st.set_page_config(page_title="Preprocessing Data", layout="wide")

# =============================================================================
# 0. AMBIL DATA TERPILIH DARI ST.SESSION_STATE
# =============================================================================
if "selected_tahun" not in st.session_state or st.session_state["selected_tahun"] is None:
    st.title("Preprocessing Data", anchor=False)
    st.warning("⚠️ Belum ada dataset yang dipilih. Silakan pilih dataset terlebih dahulu di halaman **Daftar Dataset**.")
    st.stop()

# Ambil nilai variabel dari session_state
selected_tahun = st.session_state["selected_tahun"]

# Query seluruh dataset dari database
raw_data = get_dataset()

if not raw_data:
    st.title("Preprocessing Data", anchor=False)
    st.error("❌ Data tidak ditemukan di database. Silakan upload ulang data.")
    st.stop()

df_raw_all = pd.DataFrame(raw_data)
df_raw_all.columns = [str(col).upper().strip() for col in df_raw_all.columns]

# Filter DataFrame HANYA untuk tahun yang ada di st.session_state
if 'TAHUN' in df_raw_all.columns:
    df_raw = df_raw_all[df_raw_all['TAHUN'] == selected_tahun].copy()
else:
    df_raw = df_raw_all.copy()

# Ambil Nama File Terpilih dari session_state atau dari kolom DataFrame
if "selected_file_name" in st.session_state and st.session_state["selected_file_name"]:
    nama_file_dataset = st.session_state["selected_file_name"]
elif "FILE_NAME" in df_raw.columns and pd.notnull(df_raw["FILE_NAME"].iloc[0]):
    nama_file_dataset = df_raw["FILE_NAME"].iloc[0]
elif "NAMA_FILE" in df_raw.columns and pd.notnull(df_raw["NAMA_FILE"].iloc[0]):
    nama_file_dataset = df_raw["NAMA_FILE"].iloc[0]
else:
    nama_file_dataset = f"pertanian_{selected_tahun}.csv"

# =============================================================================
# 1. PERHITUNGAN PREPROCESSING KDD
# =============================================================================

# --- TAHAP 1: CLEANING & PRODUKTIVITAS ---
df_clean = df_raw.copy()
df_clean['LUAS_PANEN'] = pd.to_numeric(df_clean['LUAS_PANEN'], errors='coerce')
df_clean['PRODUKSI'] = pd.to_numeric(df_clean['PRODUKSI'], errors='coerce')

missing_count = df_clean[['LUAS_PANEN', 'PRODUKSI']].isnull().sum().sum()
if missing_count > 0:
    df_clean = df_clean.dropna(subset=['LUAS_PANEN', 'PRODUKSI'])

df_clean['PRODUKTIVITAS'] = np.where(
    df_clean['LUAS_PANEN'] > 0, 
    df_clean['PRODUKSI'] / df_clean['LUAS_PANEN'], 
    0.0
)
df_clean['PRODUKTIVITAS'] = df_clean['PRODUKTIVITAS'].round(4)

# --- TAHAP 2: PIVOT ---
identitas_pivot = ['TAHUN', 'KECAMATAN'] if 'TAHUN' in df_clean.columns else ['KECAMATAN']

df_pivot = df_clean.pivot_table(
    index=identitas_pivot,
    columns='KOMODITAS',
    values='PRODUKTIVITAS',
    aggfunc='mean'
).fillna(0.0).reset_index()
df_pivot.columns.name = None

# --- TAHAP 3: STANDARDISASI Z-SCORE ---
identitas_cols = [col for col in identitas_pivot if col in df_pivot.columns]
komoditas_cols = [col for col in df_pivot.columns if col not in identitas_cols]

scaler = StandardScaler()
data_scaled_array = scaler.fit_transform(df_pivot[komoditas_cols])
df_scaled = pd.DataFrame(data_scaled_array, columns=komoditas_cols)

# Combined Final Matrix
df_final_preprocessed = pd.concat([df_pivot[identitas_cols], df_scaled], axis=1)

# Dimensi untuk Tampilan Visual Flowchart
total_baris_awal = len(df_raw)
total_baris_pivot = len(df_pivot)
total_fitur = len(komoditas_cols)

# =============================================================================
# 2. TAMPILAN UTAMA STREAMLIT
# =============================================================================

st.title("Preprocessing Data", anchor=False)
st.caption("Tahapan Pembersihan, Transformasi Pivot (Long-to-Wide), dan Standardisasi Z-Score")
st.markdown("---")

# -----------------------------------------------------------------------------
# A. INFORMASI DATASET TERPILIH DARI ST.SESSION_STATE
# -----------------------------------------------------------------------------
st.info(f"📄 **Dataset Terpilih:** `{nama_file_dataset}`")
st.write("")
    
# -----------------------------------------------------------------------------
# C. DETAIL TABEL HASIL PREPROCESSING (TABS)
# -----------------------------------------------------------------------------
st.subheader("🔍 Detail Hasil Preprocessing", anchor=False)

tab1, tab2, tab3, tab4 = st.tabs([
    "📥 Raw Data", 
    "🧹 Cleaning Data", 
    "🔄 Pivot", 
    "🎯 Standardisasi (Z-Score)"
])

with tab1:
    st.markdown(f"**Data Mentah Terpilih ({total_baris_awal} baris):**")
    st.dataframe(df_raw, width = "stretch", hide_index=True)

with tab2:
    st.latex(r"\text{Produktivitas (Kw/Ha)} = \frac{\text{Produksi (Kw)}}{\text{Luas Panen (Ha)}}")
    if missing_count > 0:
        st.warning(f"⚠️ Ditemukan {missing_count} baris data hilang (NaN). Baris telah dihapus.")
    else:
        st.success("✅ Audit Data Selesai: Tidak ditemukan missing value.")
    
    cols_cleaning = ['TAHUN', 'KECAMATAN', 'KOMODITAS', 'LUAS_PANEN', 'PRODUKSI', 'PRODUKTIVITAS']
    available_cols = [c for c in cols_cleaning if c in df_clean.columns]
    st.dataframe(df_clean[available_cols], use_container_width=True, hide_index=True)

with tab3:
    st.markdown(f"**Hasil Transformasi Matrix ({total_baris_pivot} baris × {total_fitur} fitur komoditas):**")
    st.dataframe(df_pivot, use_container_width=True, hide_index=True)

with tab4:
    st.latex(r"Z = \frac{X - \mu}{\sigma}")
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.markdown("**Nilai Rata-rata ($\\mu$) per Komoditas:**")
        mean_dict = {k: round(float(v), 4) for k, v in df_pivot[komoditas_cols].mean().to_dict().items()}
        st.json(mean_dict)

    with col_stat2:
        st.markdown("**Deviasi Standar ($\\sigma$) per Komoditas:**")
        std_dict = {k: round(float(v), 4) for k, v in df_pivot[komoditas_cols].std(ddof=0).to_dict().items()}
        st.json(std_dict)
        
    st.markdown("**Matrix Final Siap Masuk HDBSCAN:**")
    st.dataframe(df_final_preprocessed, use_container_width=True, hide_index=True)

st.write("")
st.markdown("---")

# -----------------------------------------------------------------------------
# D. TOMBOL SIMPAN PREPROCESSING (REDIRECT KE CLUSTERING)
# -----------------------------------------------------------------------------
col_btn1, col_btn2 = st.columns([3, 1])

with col_btn1:
    btn_simpan = st.button("💾 SIMPAN PREPROCESSING", type="primary", use_container_width=True)

with col_btn2:
    btn_reset = st.button("🔄 Reset Preprocessing", use_container_width=True)

if btn_simpan:
    with st.spinner("Menyimpan data preprocessing ke database..."):
        for col in komoditas_cols:
            df_final_preprocessed[col] = df_final_preprocessed[col].astype(float)
        
        delete_preprocessing()
        berhasil = save_preprocessing(df_final_preprocessed)

    if berhasil:
        # Simpan metadata ke session_state untuk digunakan di halaman Clustering
        st.session_state["jumlah_kecamatan"] = total_baris_pivot
        st.session_state["jumlah_fitur"] = total_fitur
        
        st.success("✅ Preprocessing Berhasil Disimpan!")
        time.sleep(0.8)
        st.switch_page("pages/5_clustering.py")
    else:
        st.error("❌ Gagal Menyimpan Data Preprocessing ke Database.")

if btn_reset:
    if delete_preprocessing():
        st.success("Data preprocessing dibersihkan.")
        time.sleep(0.8)
        st.rerun()