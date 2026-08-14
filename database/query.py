from database.connection import get_connection
import mysql.connector
import pandas as pd
import numpy as np

#LOGIN (multi-role)
def login(username, password):
    conn = get_connection()
    if conn is None:
        return None
    cursor = conn.cursor(dictionary=True)

    # Mengambil seluruh kolom termasuk 'role' dari tabel admin/user
    sql = """
    SELECT *
    FROM pengguna
    WHERE username=%s
    AND password=%s
    """
    cursor.execute(sql, (username, password))
    user = cursor.fetchone()

    cursor.close()
    conn.close()
    return user


# --- DATASET PERTANIAN (RAW DATA) ---

# Menyimpan seluruh isi DataFrame ke tabel dataset_pertanian
def insert_dataset(df):
    conn = get_connection()
    if conn is None:
        return False
    cursor = conn.cursor()

    sql = """
        INSERT INTO dataset_pertanian
        (TAHUN, KECAMATAN, KOMODITAS, LUAS_PANEN, PRODUKSI, PRODUKTIVITAS)
        VALUES (%s, %s, %s, %s, %s, %s)
    """

    for _, row in df.iterrows():
        cursor.execute(sql, (
            row["TAHUN"],
            row["KECAMATAN"],
            row["KOMODITAS"],
            row["LUAS_PANEN"],
            row["PRODUKSI"],
            row["PRODUKTIVITAS"]
        ))
    conn.commit()
    cursor.close()
    conn.close()
    return True

# Tampilkan Dataset (get_all_dataset)
def get_all_dataset():
    conn = get_connection()
    if conn is None:
        return []
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT *
        FROM dataset_pertanian
        ORDER BY id ASC
    """)
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return data

# Ambil data dari dataset (Menjaga kecocokan nama fungsi di preprocessing.py)
def get_dataset():
    return get_all_dataset()

# Jumlah Data
def total_dataset():
    conn = get_connection()
    if conn is None:
        return 0
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*)
        FROM dataset_pertanian
    """)
    total = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return total

# --- PREPROCESSING ---
# Menyimpan hasil preprocessing ke database (Per Tahun, Tidak Menimpa Tahun Lain)
def save_preprocessing(df):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # 1. Buat copy DataFrame & bersihkan nama kolom dari akhiran _ZSCORE / _zscore
        df_save = df.copy()
        df_save.columns = [
            str(col).replace('_ZSCORE', '').replace('_zscore', '').strip().replace(' ', '_') 
            for col in df_save.columns
        ]
        
        cols = list(df_save.columns)
        
        # 2. Hapus HANYA data tahun yang sedang diproses agar tidak menimpa tahun lain
        cols_upper = [c.upper() for c in cols]
        if 'TAHUN' in cols_upper:
            idx_tahun = cols_upper.index('TAHUN')
            nama_col_tahun = cols[idx_tahun]
            list_tahun = df_save[nama_col_tahun].dropna().unique().tolist()
            for t in list_tahun:
                cursor.execute("DELETE FROM preprocessing WHERE LOWER(tahun) = %s", (str(int(t)),))
        else:
            cursor.execute("TRUNCATE TABLE preprocessing")
        
        # 3. Buat query INSERT secara dinamis berdasarkan nama kolom yang sudah bersih
        columns_str = ", ".join([f"`{c.lower()}`" for c in cols])
        placeholders = ", ".join(["%s"] * len(cols))
        query = f"INSERT INTO preprocessing ({columns_str}) VALUES ({placeholders})"
        
        # 4. Iterasi setiap baris dan simpan ke database
        for _, row in df_save.iterrows():
            row_values = []
            for col in cols:
                val = row[col]
                
                # Handling NaN / Null
                if pd.isna(val):
                    row_values.append(None)
                # Handling kolom identitas
                elif col.upper() in ['TAHUN', 'KECAMATAN']:
                    row_values.append(str(val))
                # Handling nilai numerik / Z-Score (Konversi float64 -> float murni Python)
                else:
                    row_values.append(float(val))
            
            cursor.execute(query, tuple(row_values))
            
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving preprocessing: {e}")
        return False

# Menampilkan hasil preprocessing
def get_preprocessing():
    conn = get_connection()
    if conn is None:
        return []
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT *
        FROM preprocessing
        ORDER BY id ASC
    """)
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return data

# Hapus hasil preprocessing
def delete_preprocessing():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("TRUNCATE TABLE preprocessing")
    conn.commit()
    cursor.close()
    conn.close()
    return True

# CLUSTERING HDBSCAN
# Menyimpan hasil clustering ke database (Per Tahun, Tidak Menimpa Tahun Lain)
def save_hasil_clustering(df_hasil):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # 1. Buat copy DataFrame & bersihkan nama kolom
        df_save = df_hasil.copy()
        df_save.columns = [
            str(col).replace('_ZSCORE', '').replace('_zscore', '').strip().lower() 
            for col in df_save.columns
        ]
        
        cols = list(df_save.columns)
        
        # 2. Hapus HANYA hasil tahun yang sedang diproses agar data tahun lain tetap tersimpan
        if 'tahun' in cols:
            list_tahun = df_save['tahun'].dropna().unique().tolist()
            for t in list_tahun:
                cursor.execute("DELETE FROM hasil_clustering WHERE tahun = %s", (int(t),))
        else:
            cursor.execute("TRUNCATE TABLE hasil_clustering")
        
        # 3. Buat query SQL dinamis
        columns_str = ", ".join([f"`{c}`" for c in cols])
        placeholders = ", ".join(["%s"] * len(cols))
        query = f"INSERT INTO hasil_clustering ({columns_str}) VALUES ({placeholders})"
        
        # 4. Iterasi baris dan konversi tipe data numpy ke Python native
        for _, row in df_save.iterrows():
            row_values = []
            for col in cols:
                val = row[col]
                
                if pd.isna(val):
                    row_values.append(None)
                elif isinstance(val, (int, np.integer)):
                    row_values.append(int(val))
                elif isinstance(val, (float, np.floating)):
                    row_values.append(float(val))
                else:
                    row_values.append(str(val))
                    
            cursor.execute(query, tuple(row_values))
            
        conn.commit()
        cursor.close()
        conn.close()
        return True, "Berhasil"
        
    except Exception as e:
        print(f"Error saving hasil clustering: {e}")
        return False, str(e)

# Menampilkan hasil clustering
def get_hasil_clustering():
    conn = get_connection()
    if conn is None:
        return []

    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT *
        FROM hasil_clustering
        ORDER BY id ASC
    """)

    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return data

# Hapus hasil clustering
def delete_hasil_clustering():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("TRUNCATE TABLE hasil_clustering")
    conn.commit()
    cursor.close()
    conn.close()

    return True

# Hapus seluruh data secara berantai (Cascading Delete via Python)
def delete_all_dataset():
    """
    Menghapus seluruh baris data dari ketiga tabel secara pasti.
    """
    conn = get_connection()
    if conn is None:
        return False
        
    try:
        cursor = conn.cursor()
        
        # Matikan FK Check agar tidak terhalang constraint
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
        
        # Hapus isi semua tabel satu persatu
        cursor.execute("DELETE FROM hasil_clustering;")
        cursor.execute("DELETE FROM preprocessing;")
        cursor.execute("DELETE FROM dataset_pertanian;")
        
        # Reset AUTO_INCREMENT id ke 1 kembali
        cursor.execute("ALTER TABLE hasil_clustering AUTO_INCREMENT = 1;")
        cursor.execute("ALTER TABLE preprocessing AUTO_INCREMENT = 1;")
        cursor.execute("ALTER TABLE dataset_pertanian AUTO_INCREMENT = 1;")
        
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error saat menghapus data: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return False