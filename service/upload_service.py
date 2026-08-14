import pandas as pd
import streamlit as st

def read_excel(file):
    """
    Fungsi untuk membaca file Excel yang diunggah dari Streamlit
    dan mengubahnya menjadi Pandas DataFrame.
    """
    try:
        # Membaca excel menggunakan pandas
        df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Gagal membaca file Excel: {e}")
        return None