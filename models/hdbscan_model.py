import pandas as pd
import numpy as np
from sklearn.cluster import HDBSCAN

def run_hdbscan(df, min_cluster_size=5, min_samples=5):
    """
    Menjalankan proses clustering HDBSCAN berdasarkan data hasil pivot/preprocessing.
    """
    # Copy dataframe agar data asli tidak berubah
    hasil = df.copy()

    # 1. Tentukan nama kolom identitas/metadata yang BUKAN fitur clustering
    non_feature_cols = ['ID', 'ID_PREPROCESSING', 'TAHUN', 'KECAMATAN', 'CLUSTER', 'PROBABILITY', 'OUTLIER_SCORE']
    
    # 2. Ambil seluruh kolom fitur komoditas (Z-Score) secara otomatis/dinamis
    fitur_cols = [col for col in hasil.columns if col.upper() not in non_feature_cols]

    # Ambil matriks data X
    X = hasil[fitur_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    # 3. Proteksi jika data kosong atau min_samples melebihi total baris data
    n_samples = len(X)
    if n_samples == 0:
        raise ValueError("Data masukan X kosong! Silakan periksa kembali data di halaman preprocessing.")

    # Mencegah error jika min_samples/min_cluster_size lebih besar dari jumlah data
    actual_min_samples = min(int(min_samples), n_samples)
    actual_min_cluster_size = min(int(min_cluster_size), n_samples)

    # 4. Inisialisasi dan Jalankan HDBSCAN
    clusterer = HDBSCAN(
        min_cluster_size=actual_min_cluster_size,
        min_samples=actual_min_samples,
        metric='euclidean'
    )
    
    labels = clusterer.fit_predict(X)

    # 5. Simpan hasil clustering
    hasil["CLUSTER"] = labels

    # Hitung nilai Probabilitas & Outlier Score
    if hasattr(clusterer, 'probabilities_'):
        probs = clusterer.probabilities_
        hasil["PROBABILITY"] = np.round(probs, 4)
        # Data noise (probability 0) akan bernilai OUTLIER_SCORE = 1.0
        hasil["OUTLIER_SCORE"] = np.round(1.0 - probs, 4)
    else:
        hasil["PROBABILITY"] = 1.0
        hasil["OUTLIER_SCORE"] = 0.0

    return hasil