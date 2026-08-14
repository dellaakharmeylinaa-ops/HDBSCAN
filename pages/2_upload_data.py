import numpy as np
import pandas as pd
import streamlit as st
import time  # jeda sebelum pindah halaman
from service.upload_service import read_excel
from database.query import insert_dataset

st.title("UPLOAD DATA", anchor=False)
st.write("Unggah data produktivitas hasil pertanian dalam format Excel")
st.markdown("---")
col1, col2 = st.columns([2, 2])

with col1:
    uploaded_file = st.file_uploader(
        "Pilih File Excel",
        type=["xlsx", "xls"],
        label_visibility="collapsed"
    )

with col2:
    if uploaded_file is None:
        st.info("Belum ada file yang dipilih.")
    else:
        ukuran = uploaded_file.size / 1024
        st.success("File berhasil dipilih")
        st.write(f"**Nama File :** {uploaded_file.name}")
        st.write(f"**Ukuran :** {ukuran:.2f} KB")
        st.progress(100)

if uploaded_file is not None:
    df = read_excel(uploaded_file)
    
    # 1. Bersihkan nama kolom
    df.columns = df.columns.str.strip().str.upper().str.replace(' ', '_')
    
    # 2. Bersihkan dan konversi kolom angka (PRODUKSI, PRODUKTIVITAS, LUAS_PANEN, dll)
    kolom_angka = [col for col in ['PRODUKSI', 'PRODUKTIVITAS', 'LUAS_PANEN', 'LUAS_TANAM'] if col in df.columns]
    
    def parse_angka_aman(val):
        if pd.isna(val):
            return None
        if isinstance(val, (int, float, np.number)):
            return float(val)
        
        s = str(val).strip()
        if s in ['', '-', 'None', 'nan', 'NaN', 'null']:
            return None
            
        # Jika ada koma dan titik (misal 1.250,50), titik adalah ribuan dan koma adalah desimal
        if ',' in s and '.' in s:
            s = s.replace('.', '').replace(',', '.')
        # Jika hanya ada koma (misal 12,5 atau 1250,5), koma adalah desimal
        elif ',' in s:
            s = s.replace(',', '.')
        # Jika hanya ada titik, biarkan sebagai desimal (misal 12.5)
        
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    for col in kolom_angka:
        df[col] = df[col].apply(parse_angka_aman)
    
    # 3. Ubah nilai NaN menjadi None agar tersimpan NULL di MySQL
    df = df.replace({np.nan: None})
    # ====================================================================

    st.markdown("---")
    st.subheader("Preview Data Excel")
    st.dataframe(
        df,
        width="stretch",
        hide_index=True
    )

st.markdown("---")
col1, col2, col3 = st.columns([5, 1, 1])

with col1:
    st.info("Pastikan data sudah sesuai sebelum disimpan ke dalam database!.")

with col2:
    if st.button("Batal", width="stretch"):
        st.rerun()

with col3:
    simpan = st.button("Simpan Data", width="stretch")

if simpan:
    if uploaded_file is None:
        st.error("Silahkan Pilih File Terlebih Dahulu.")
    else:
        berhasil = insert_dataset(df)
        if berhasil:
            st.success("Data Tersimpan ke database.")
            time.sleep(1.5)  # Beri jeda 1.5 detik agar user sempat melihat pesan sukses
            st.switch_page("pages/3_daftar_dataset.py")
        else:
            st.error("Data Gagal Disimpan.")