"""
================================================================
  UTILS.PY — Data Loading & Preprocessing (VERSI BPS)
  Khusus untuk format Excel BPS (Wide/Pivot Format)
================================================================
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def load_data(file_path: str) -> pd.DataFrame:
    """
    Memuat file Excel BPS dengan format WIDE (pivot table).
    Struktur file:
    - Baris 0-1: Judul (merge)
    - Baris 2: Nama komoditas (Kelapa Sawit, Kelapa, Karet, dll)
    - Baris 3: Tahun (2024)
    - Baris 4+: Data provinsi
    """
    try:
        # Baca Excel tanpa header dulu untuk deteksi struktur
        df_raw = pd.read_excel(file_path, header=None)
        
        # Cari baris yang berisi nama komoditas (biasanya baris ke-3 atau ke-4)
        header_row = None
        for i in range(min(10, len(df_raw))):
            row_values = [str(v).lower() for v in df_raw.iloc[i].values if pd.notna(v)]
            if any("kelapa" in v or "karet" in v or "kopi" in v for v in row_values):
                header_row = i
                break
        
        if header_row is None:
            header_row = 2  # Default fallback
        
        # Baca ulang dengan header yang benar
        df = pd.read_excel(file_path, header=header_row)
        
        return df
        
    except Exception as e:
        raise ValueError(f"Gagal memuat file Excel: {e}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mengubah format WIDE menjadi LONG (tidy format) dan membersihkan data.
    
    Proses:
    1. Identifikasi kolom provinsi (kolom pertama)
    2. Identifikasi kolom komoditas (kolom 2+)
    3. Unpivot/melt dari wide ke long format
    4. Konversi Ribu Ton ke Ton (x1000)
    5. Hapus baris total (INDONESIA)
    6. Handle missing values
    """
    df = df.copy()
    
    # ===== 1. IDENTIFIKASI KOLOM =====
    # Kolom pertama biasanya adalah Provinsi
    first_col = df.columns[0]
    
    # Rename kolom pertama jadi "Provinsi"
    df = df.rename(columns={first_col: "Provinsi"})
    
    # ===== 2. HAPUS BARIS TOTAL & HEADER BERLEBIH =====
    # Hapus baris yang berisi "INDONESIA", "38 Provinsi", atau angka
    keywords_to_remove = ["indonesia", "38 provinsi", "jumlah", "total", "nan"]
    mask = df["Provinsi"].astype(str).str.lower().str.contains(
        "|".join(keywords_to_remove), na=False, regex=True
    )
    df = df[~mask]
    
    # Hapus baris dengan provinsi kosong
    df = df[df["Provinsi"].notna() & (df["Provinsi"].astype(str).str.strip() != "")]
    
    # ===== 3. IDENTIFIKASI KOLOM KOMODITAS =====
    # Semua kolom selain "Provinsi" adalah komoditas
    commodity_cols = [c for c in df.columns if c != "Provinsi"]
    
    # Filter hanya kolom yang benar-benar berisi nama komoditas
    valid_commodities = []
    for col in commodity_cols:
        col_str = str(col).strip()
        # Skip kolom yang bukan komoditas (tahun, angka, kosong)
        if col_str and not col_str.replace(".", "").isdigit() and col_str != "nan":
            # Skip jika kolom adalah tahun (4 digit angka)
            if not (len(col_str) == 4 and col_str.isdigit()):
                valid_commodities.append(col)
    
    if not valid_commodities:
        raise ValueError("Tidak dapat menemukan kolom komoditas dalam file Excel.")
    
    # ===== 4. DETEKSI TAHUN =====
    year = 2024  # Default
    for col in commodity_cols:
        col_str = str(col).strip()
        if len(col_str) == 4 and col_str.isdigit():
            year = int(col_str)
            break
    
    # ===== 5. MELT (UNPIVOT) DATA DARI WIDE KE LONG =====
    df_long = df[["Provinsi"] + valid_commodities].copy()
    df_long = df_long.melt(
        id_vars=["Provinsi"],
        value_vars=valid_commodities,
        var_name="Komoditas",
        value_name="Produksi (Ribu Ton)"
    )
    
    # ===== 6. KONVERSI KE TON =====
    # Data BPS dalam Ribu Ton, konversi ke Ton
    df_long["Produksi (Ton)"] = pd.to_numeric(
        df_long["Produksi (Ribu Ton)"], 
        errors="coerce"
    ) * 1000
    
    # ===== 7. CLEANING NILAI =====
    # Handle nilai negatif atau NaN
    df_long["Produksi (Ton)"] = df_long["Produksi (Ton)"].fillna(0)
    df_long["Produksi (Ton)"] = df_long["Produksi (Ton)"].clip(lower=0)
    
    # Tambahkan kolom Tahun
    df_long["Tahun"] = year
    
    # Normalisasi nama provinsi (UPPERCASE & strip)
    df_long["Provinsi"] = df_long["Provinsi"].astype(str).str.strip().str.upper()
    df_long["Komoditas"] = df_long["Komoditas"].astype(str).str.strip().str.title()
    
    # ===== 8. TAMBAH KOLOM TAMBAHAN (ESTIMASI) =====
    # Karena data asli hanya berisi Produksi, kita bisa tambahkan estimasi
    # untuk visualisasi yang lebih menarik
    
    # Estimasi Luas (Ha) berdasarkan rata-rata produktivitas per komoditas
    # (Ini estimasi untuk keperluan visualisasi saja)
    produktivitas_rata_rata = {
        "Kelapa Sawit": 3.5,   # ton/ha
        "Kelapa": 1.2,
        "Karet": 1.5,
        "Kopi": 0.8,
        "Kakao": 0.9,
        "Teh": 2.0,
        "Tebu": 7.0,
    }
    
    df_long["Produktivitas (Ton/Ha)"] = df_long["Komoditas"].map(produktivitas_rata_rata).fillna(1.0)
    df_long["Estimasi Luas (Ha)"] = df_long["Produksi (Ton)"] / df_long["Produktivitas (Ton/Ha)"]
    
    # ===== 9. SORT & RESET INDEX =====
    df_long = df_long.sort_values(
        by=["Provinsi", "Komoditas"], 
        ascending=[True, True]
    ).reset_index(drop=True)
    
    # Pilih kolom final
    final_columns = [
        "Provinsi", 
        "Komoditas", 
        "Produksi (Ton)",
        "Estimasi Luas (Ha)",
        "Produktivitas (Ton/Ha)",
        "Tahun"
    ]
    
    return df_long[final_columns]


def get_summary_stats(df: pd.DataFrame, production_col: str) -> pd.DataFrame:
    """
    Menghasilkan ringkasan statistik deskriptif dari data.
    """
    if production_col not in df.columns:
        return pd.DataFrame({"Error": ["Kolom produksi tidak ditemukan"]})
    
    stats = {
        "Jumlah Data": len(df),
        "Total Produksi (Ton)": df[production_col].sum(),
        "Rata-rata Produksi": df[production_col].mean(),
        "Median Produksi": df[production_col].median(),
        "Standar Deviasi": df[production_col].std(),
        "Produksi Minimum": df[production_col].min(),
        "Produksi Maksimum": df[production_col].max(),
        "Q1 (25%)": df[production_col].quantile(0.25),
        "Q3 (75%)": df[production_col].quantile(0.75),
        "Total Missing": df[production_col].isna().sum(),
    }
    
    # Tambahkan info komoditas
    if "Komoditas" in df.columns:
        top_commodity = (
            df.groupby("Komoditas")[production_col]
            .sum()
            .sort_values(ascending=False)
            .index[0]
        )
        stats["Komoditas Teratas"] = top_commodity
    
    # Tambahkan info provinsi
    if "Provinsi" in df.columns:
        top_province = (
            df.groupby("Provinsi")[production_col]
            .sum()
            .sort_values(ascending=False)
            .index[0]
        )
        stats["Provinsi Teratas"] = top_province
    
    return pd.DataFrame.from_dict(stats, orient="index", columns=["Nilai"])
