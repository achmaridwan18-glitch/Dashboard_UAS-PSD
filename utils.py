"""
================================================================
  UTILS.PY — Data Loading & Preprocessing
  Dashboard Produksi Tanaman Perkebunan Indonesia
================================================================
"""

import pandas as pd
import numpy as np
import re
import warnings

warnings.filterwarnings("ignore")


def load_data(file_path: str) -> pd.DataFrame:
    """
    Memuat file Excel dari BPS dengan handling multi-header.
    Data BPS biasanya memiliki 3-5 baris header sebelum data sebenarnya.
    """
    try:
        # Coba baca dengan berbagai strategi
        # Strategi 1: Header di baris ke-0 (format standar)
        df = pd.read_excel(file_path)

        # Jika kolom pertama terlihat aneh (berisi angka atau NaN),
        # kemungkinan header ada di baris bawahnya
        if df.columns[0] is None or str(df.columns[0]).strip() == "":
            df = pd.read_excel(file_path, header=1)

        # Deteksi otomatis baris header BPS
        # Biasanya ada kata "Provinsi", "Tanaman", "Produksi" di salah satu baris
        for skip in range(0, 8):
            df_test = pd.read_excel(file_path, header=skip)
            cols_lower = [str(c).lower() for c in df_test.columns]
            if any("provinsi" in c or "prov" in c for c in cols_lower):
                df = df_test
                break

        return df

    except Exception as e:
        raise ValueError(f"Gagal memuat file Excel: {e}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan data BPS:
    - Rename kolom agar konsisten
    - Hapus baris total/subtotal/judul
    - Konversi kolom numerik
    - Handle missing values
    """
    df = df.copy()

    # ===== 1. NORMALISASI NAMA KOLOM =====
    # Rename kolom berdasarkan pola umum BPS
    column_mapping = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()

        if "provinsi" in col_lower or col_lower == "prov":
            column_mapping[col] = "Provinsi"
        elif any(k in col_lower for k in ["komoditas", "jenis tanaman", "jenis komoditas", "tanaman"]):
            column_mapping[col] = "Komoditas"
        elif "produksi" in col_lower and "ton" in col_lower:
            column_mapping[col] = "Produksi (Ton)"
        elif "produksi" in col_lower and "kuintal" in col_lower:
            column_mapping[col] = "Produksi (Kuintal)"
        elif "produksi" in col_lower:
            column_mapping[col] = "Produksi (Ton)"
        elif "luas" in col_lower:
            column_mapping[col] = "Luas (Ha)"
        elif "produktivitas" in col_lower or "rendemen" in col_lower:
            column_mapping[col] = "Produktivitas (Ton/Ha)"
        elif "tahun" in col_lower:
            column_mapping[col] = "Tahun"

    df = df.rename(columns=column_mapping)

    # ===== 2. HAPUS BARIS TOTAL / SUBTOTAL / CATATAN =====
    if "Provinsi" in df.columns:
        keywords_to_remove = [
            "indonesia", "jumlah", "total", "sum", "nanggroe",
            "catatan", "note", "source", "sumber", "nan"
        ]
        mask = df["Provinsi"].astype(str).str.lower().str.contains(
            "|".join(keywords_to_remove), na=False, regex=True
        )
        df = df[~mask]

        # Hapus baris dengan provinsi kosong
        df = df[df["Provinsi"].notna() & (df["Provinsi"].astype(str).str.strip() != "")]

    # ===== 3. KONVERSI KOLOM NUMERIK =====
    numeric_cols = ["Produksi (Ton)", "Produksi (Kuintal)", "Luas (Ha)", "Produktivitas (Ton/Ha)"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(".", "", regex=False)
                .str.replace("-", "0", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Jika kolom Produksi (Ton) tidak ada tapi ada Kuintal, konversi
    if "Produksi (Ton)" not in df.columns and "Produksi (Kuintal)" in df.columns:
        df["Produksi (Ton)"] = df["Produksi (Kuintal)"] / 10

    # ===== 4. HANDLE MISSING VALUES =====
    # Drop baris yang tidak punya data penting
    essential_cols = ["Provinsi", "Komoditas"]
    existing_essential = [c for c in essential_cols if c in df.columns]
    if existing_essential:
        df = df.dropna(subset=existing_essential)

    # ===== 5. RESET INDEX =====
    df = df.reset_index(drop=True)

    # ===== 6. TAMBAH KOLOM TAHUN (jika tidak ada) =====
    if "Tahun" not in df.columns:
        df["Tahun"] = 2025

    return df


def get_summary_stats(df: pd.DataFrame, production_col: str) -> pd.DataFrame:
    """
    Menghasilkan ringkasan statistik deskriptif dari data.
    """
    if production_col not in df.columns:
        return pd.DataFrame({"Error": ["Kolom produksi tidak ditemukan"]})

    stats = {
        "Jumlah Data": len(df),
        "Total Produksi": df[production_col].sum(),
        "Rata-rata": df[production_col].mean(),
        "Median": df[production_col].median(),
        "Standar Deviasi": df[production_col].std(),
        "Minimum": df[production_col].min(),
        "Maksimum": df[production_col].max(),
        "Q1 (25%)": df[production_col].quantile(0.25),
        "Q3 (75%)": df[production_col].quantile(0.75),
        "Total Missing": df[production_col].isna().sum(),
    }

    # Tambahkan statistik per komoditas jika ada
    if "Komoditas" in df.columns:
        top_commodity = (
            df.groupby("Komoditas")[production_col]
            .sum()
            .sort_values(ascending=False)
            .index[0]
        )
        stats["Komoditas Teratas"] = top_commodity

    if "Provinsi" in df.columns:
        top_province = (
            df.groupby("Provinsi")[production_col]
            .sum()
            .sort_values(ascending=False)
            .index[0]
        )
        stats["Provinsi Teratas"] = top_province

    return pd.DataFrame.from_dict(stats, orient="index", columns=["Nilai"])


def detect_columns(df: pd.DataFrame) -> dict:
    """
    Mendeteksi kolom-kolom penting secara otomatis.
    Berguna jika nama kolom tidak standar.
    """
    detected = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if "provinsi" in col_lower or col_lower == "prov":
            detected["province"] = col
        elif any(k in col_lower for k in ["komoditas", "tanaman", "jenis"]):
            detected["commodity"] = col
        elif "produksi" in col_lower:
            detected["production"] = col
        elif "luas" in col_lower:
            detected["area"] = col
        elif "produktivitas" in col_lower:
            detected["productivity"] = col
        elif "tahun" in col_lower:
            detected["year"] = col

    return detected
