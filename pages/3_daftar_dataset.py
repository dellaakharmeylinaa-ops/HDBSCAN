import streamlit as st
import pandas as pd
import time
from database.query import (
    get_all_dataset, 
    total_dataset, 
    delete_all_dataset,
    get_preprocessing,
    get_hasil_clustering  # Imported untuk mengecek status pemrosesan clustering
)

st.set_page_config(page_title="Daftar Dataset", layout="wide")
st.title("DAFTAR DATASET")
st.write("Menampilkan daftar file dataset yang telah di-upload. Pilih dataset aktif yang akan digunakan untuk proses Preprocessing dan Clustering.")
st.markdown("---")

# -----------------------------------------------------------------------------
# AMBIL DATA DARI DATABASE
# -----------------------------------------------------------------------------
data_raw = get_all_dataset()
data_clustering = get_hasil_clustering() # Ambil hasil clustering untuk validasi status

# Konversi data mentah ke DataFrame
if data_raw:
    df = pd.DataFrame(data_raw)
    df.columns = [str(col).upper().strip() for col in df.columns]
else:
    df = pd.DataFrame()

# Konversi data clustering ke DataFrame
if data_clustering:
    df_clustering = pd.DataFrame(data_clustering)
    df_clustering.columns = [str(col).upper().strip() for col in df_clustering.columns]
else:
    df_clustering = pd.DataFrame()


# =============================================================================
# 1. RINGKASAN FILE DATASET & PEMILIHAN DATASET DIPROSES
# =============================================================================
if not df.empty:
    # Cek ketersediaan kolom TAHUN untuk pengelompokan file dataset
    if "TAHUN" in df.columns:
        list_tahun = sorted(df["TAHUN"].unique().tolist(), reverse=True)
        
        # Inisialisasi session state untuk dataset yang dipilih (default: tahun terbaru)
        if "selected_tahun" not in st.session_state or st.session_state["selected_tahun"] not in list_tahun:
            st.session_state["selected_tahun"] = list_tahun[0]

        # ---------------------------------------------------------------------
        # TABEL RINGKASAN DATASET
        # ---------------------------------------------------------------------
        summary_list = []
        for idx, tahun in enumerate(list_tahun, start=1):
            group = df[df["TAHUN"] == tahun]
            
            # Mendapatkan Nama File yang Diunggah
            if "FILE_NAME" in group.columns and pd.notnull(group["FILE_NAME"].iloc[0]):
                nama_file = group["FILE_NAME"].iloc[0]
            elif "NAMA_FILE" in group.columns and pd.notnull(group["NAMA_FILE"].iloc[0]):
                nama_file = group["NAMA_FILE"].iloc[0]
            else:
                nama_file = f"data penelitian {int(tahun)}.xls"
            
            # Pengecekan Status: "Sudah diproses" HANYA jika hasil clustering untuk tahun ini sudah ada
            has_clustering = False
            if not df_clustering.empty:
                if "TAHUN" in df_clustering.columns:
                    has_clustering = len(df_clustering[df_clustering["TAHUN"] == tahun]) > 0
                else:
                    # Jika tabel clustering tidak memiliki kolom TAHUN, berasumsi sudah ada data hasil
                    has_clustering = True
            
            status_label = "Sudah diproses" if has_clustering else "Belum diproses"
            
            summary_list.append({
                "ID": idx,
                "Nama Dataset": nama_file,
                "Tahun": int(tahun) if pd.notnull(tahun) else "-",
                "Jumlah Data": len(group),
                "Status": status_label
            })
            
        df_summary = pd.DataFrame(summary_list)
        st.dataframe(df_summary, width="stretch", hide_index=True)

        st.write("")

        with st.container(border=True):
            st.markdown("📌 **Pilih Dataset yang Akan Diproses Ke Preprocessing:**")
            
            col_select, col_btn = st.columns([3, 1])
            
            # Pilihan nama file berbasis data dari summary_list
            file_options = {
                row["Tahun"]: f"{row['Nama Dataset']}"
                for row in summary_list
            }
            
            with col_select:
                tahun_terpilih = st.selectbox(  
                    "Pilih File Dataset:",
                    options=list(file_options.keys()),
                    format_func=lambda x: file_options[x],
                    index=list(file_options.keys()).index(st.session_state["selected_tahun"]) if st.session_state["selected_tahun"] in file_options else 0,
                    label_visibility="collapsed"
                )
            
            with col_btn:
                # Simpan status klik ke dalam variabel btn_proses
                btn_proses = st.button("🚀 Proses Dataset", type="primary", width="stretch", key="btn_proses_summary")

            if btn_proses:
                st.session_state["selected_tahun"] = tahun_terpilih
                st.success(f"✅ Memproses {file_options[tahun_terpilih]}...")
                time.sleep(0.8)  # Diberikan sedikit waktu agar animasi/notifikasi terlihat jelas
                st.switch_page("pages/4_preprocessing.py")

    else:
        # Jika tidak ada kolom TAHUN (Hanya 1 file dataset)
        if "FILE_NAME" in df.columns and pd.notnull(df["FILE_NAME"].iloc[0]):
            nama_file = df["FILE_NAME"].iloc[0]
        else:
            nama_file = "data penelitian 2023.xls"
        
        has_clustering = not df_clustering.empty
        status_label = "Sudah diproses" if has_clustering else "Belum diproses"

        df_summary = pd.DataFrame([{
            "ID": 1,
            "Nama Dataset": nama_file,
            "Tahun": "-",
            "Jumlah Data": len(df),
            "Status": status_label,
            "Pilihan Diproses": "✅ AKSI / DIPILIH"
        }])
        st.dataframe(df_summary, width='stretch', hide_index=True)
        st.session_state["selected_tahun"] = None

    # =============================================================================
    # 2. ZONA BAHAYA / AKSI HAPUS DATASET
    # =============================================================================
    st.write("")
    st.markdown("---")
    
    col_info, col_delete = st.columns([3, 1])
    
    with col_info:
        st.markdown("⚠️ **Pengaturan Data:** Hapus seluruh isi dataset dan database jika ingin mengunggah dataset baru dari awal.")
    
    with col_delete:
        # Menggunakan Popover sebagai dialog konfirmasi aman
        with st.popover("🗑️ Hapus Semua Dataset", width="stretch"):
            st.warning("⚠️ **Tindakan ini tidak dapat dibatalkan!**")
            st.write("Apakah Anda yakin ingin menghapus seluruh isi dataset dan database?")
            
            btn_confirm_delete = st.button("Ya, Hapus Sekarang", type="primary", key="confirm_delete_all", width="stretch")
            
            if btn_confirm_delete:
                try:
                    # Memanggil fungsi penghapusan data dari database/query.py
                    delete_all_dataset()
                    
                    # Bersihkan session state terkait dataset
                    st.session_state.pop("selected_tahun", None)
                    
                    st.success("✅ Seluruh dataset berhasil dihapus dari database!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Gagal menghapus dataset: {e}")

else:
    st.info("ℹ️ Belum ada file dataset yang di-upload. Silakan lakukan upload data terlebih dahulu.")