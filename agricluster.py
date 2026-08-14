import streamlit as st
from database.query import login

#konfigurasi Halaman (Harus berada di paling atas)
st.set_page_config(
    page_title="AGRICLUSTER",
    page_icon="🌾",
    layout="wide"
)

#inisialisasi session state di awal
if "login" not in st.session_state:
    st.session_state.login = False
if "user" not in st.session_state:
    st.session_state.user = None

# fungsi utk tampilan login
def show_login_page():
    # Spasi vertikal agar tampilan agak turun ke bawah
    st.write("")
    st.write("")
    
    # Gunakan SATU set kolom saja agar judul dan form sejajar sempurna
    left_co, cent_co, right_co = st.columns([1, 2, 1])
    
    with cent_co:
        # Tulis judul di sini (di luar form, tapi tetap di dalam kolom tengah)
        st.title("Login Pengguna", anchor=False)
        
        # Baru kemudian gambar kotak form di bawahnya
        with st.form("Login Form"):
            username = st.text_input("Username", placeholder="Masukkan username...")
            password = st.text_input("Password", type="password", placeholder="Masukkan password...")
            submit_button = st.form_submit_button("Login", width = "stretch", type = "primary")

            if submit_button:
                if not username or not password:
                    st.warning("Username dan Password wajib diisi.")
                else:
                    user = login(username, password)
                    if user:
                        st.session_state.login = True
                        st.session_state.user = user
                        st.success("Login Berhasil!")
                        st.rerun()
                    else:
                        st.error("Username atau Password salah.")

#definisikan halaman aplikasi
#halaman login
login_page = st.Page(show_login_page, title = "Login Pengguna")

#halaman bersama (admin $ pimpinan)
dashboard_page = st.Page("pages/1_dashboard.py", title = "Dashboard")
hasil_page = st.Page("pages/6_hasil_cluster.py", title = "Hasil Clustering")
visualisasi_page = st.Page("pages/7_visualisasi.py", title = "Visualisasi & Laporan")

#halaman khusus admin dkpp
upload_page = st.Page("pages/2_upload_data.py", title = "Upload Data")
dataset_page = st.Page("pages/3_daftar_dataset.py", title = "Daftar Dataset")
preprocessing_page = st.Page("pages/4_preprocessing.py", title = "Preprocessing Data")
clustering_page = st.Page("pages/5_clustering.py", title = "Clustering HDBSCAN")

#LOGIKA NAVIGASI DENGAN ROLE-BASED ACCESS
#BELUM LOGIN (Tampilkan Form Login)
if not st.session_state.login:
    #posisi navigasi sidebar tidak muncul
    pg = st.navigation([login_page], position = "hidden")
    pg.run()

#SUDAH LOGIN (Tampilkan Dashboard Utama)
else:
    #ambil role dari session_state user (pastikan di query login SELECT role juga)
    role = st.session_state.user.get("role", "pimpinan").lower()

    if role == "admin":
        #hak akses admin dkpp
        pg = st.navigation({
            "": [dashboard_page],
            "Kelola Data": [
                upload_page,
                dataset_page,
                preprocessing_page,
                clustering_page,
            ],
            "Laporan & Analisis": [
                hasil_page,
                visualisasi_page,
            ]
        })
    else:
        # hak akses Pimpinan (Viewer)
        # Hanya bisa melihat Dashboard, Hasil Clustering, dan Visualisasi/Laporan
        pg = st.navigation({
            "": [dashboard_page],
            "Laporan Exekutif":[
                hasil_page,
                visualisasi_page,
            ]
        })
   
    #sidebar informasi user & tombol logout
    with st.sidebar:
        st.markdown("---")
        username_display = st.session_state.user.get('username', 'User')
        role_display = role.upper()

        st.write(f"Login sebagai: **{username_display}**")
        
        #tombol logout merah (pakai tipe primary agar terlihat jelas)
        if st.button("Logout", type = "primary", use_container_width = True):
            st.session_state.login = False
            st.session_state.user = None
            st.success("Logout Berhasil!")
            st.rerun()
    #jalankan halaman terpilih
    pg.run()