import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN

def transform_and_scale(df):
    """
    Melakukan pivot table long-to-wide, memisahkan identitas, 
    dan melakukan standardisasi Z-score pada fitur produktivitas.
    """
    # 1. Pivot Data: Ubah Komoditas menjadi kolom terpisah
    df_pivot = df.pivot_table(
        index=["TAHUN", "KECAMATAN"],
        columns="KOMODITAS",
        values="PRODUKTIVITAS",
        aggfunc="mean"
    ).fillna(0).reset_index()

    # 2. Separasi Identitas & Fitur Fitur
    identitas = df_pivot[["TAHUN", "KECAMATAN"]].copy()
    
    # Ambil semua kolom komoditas
    fitur_cols = [col for col in df_pivot.columns if col not in ["TAHUN", "KECAMATAN"]]
    X_raw = df_pivot[fitur_cols].copy()

    # 3. Standardisasi Z-Score
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X_raw)
    X_scaled = pd.DataFrame(X_scaled_array, columns=fitur_cols)

    return identitas, X_raw, X_scaled, df_pivot


def run_hdbscan(X_scaled, min_cluster_size=3, min_samples=None, metric="euclidean"):
    """
    Menjalankan algoritma HDBSCAN pada data yang sudah di-scale.
    """
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples if min_samples and min_samples > 0 else None,
        metric=metric
    )
    
    # Fit dan Predict
    labels = model.fit_predict(X_scaled)
    
    # Ambil probabilitas keanggotaan kluster (jika tersedia)
    probabilities = getattr(model, "probabilities_", np.ones(len(labels)))

    return labels, probabilities, model