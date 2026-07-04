"""
================================================================
  DASHBOARD PRODUKSI TANAMAN PERKEBUNAN INDONESIA
  UAS Data Science — Modern & Professional Dashboard
  Author: [Nama Anda]
  Tanggal: Juli 2026
================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import base64
import io

from utils import load_data, clean_data, get_summary_stats
from model import train_random_forest, train_linear_regression, get_feature_importance

warnings.filterwarnings("ignore")

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Dashboard Perkebunan Indonesia",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. CUSTOM CSS — MODERN GREEN GLASSMORPHISM THEME
# ============================================================
def load_css():
    st.markdown("""
    <style>
    /* ---------- GLOBAL BACKGROUND ---------- */
    .stApp {
        background: linear-gradient(135deg, #f0f9f0 0%, #e8f5e9 100%);
    }

    /* ---------- HEADER ---------- */
    .main-header {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #43a047 100%);
        padding: 25px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(46, 125, 50, 0.25);
    }
    .main-header h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
    }
    .main-header p {
        margin: 5px 0 0 0;
        opacity: 0.9;
        font-size: 14px;
    }

    /* ---------- KPI CARDS ---------- */
    .kpi-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(46, 125, 50, 0.2);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(46, 125, 50, 0.18);
    }
    .kpi-icon {
        font-size: 32px;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #1b5e20;
        margin: 4px 0;
    }
    .kpi-label {
        font-size: 13px;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ---------- SECTION TITLE ---------- */
    .section-title {
        color: #1b5e20;
        font-size: 20px;
        font-weight: 700;
        border-left: 5px solid #43a047;
        padding-left: 12px;
        margin: 20px 0 12px 0;
    }

    /* ---------- INSIGHT BOX ---------- */
    .insight-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 5px solid #2e7d32;
        padding: 16px 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: #1b5e20;
    }
    .insight-box strong { color: #1b5e20; }

    /* ---------- SIDEBAR ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f1f8e9 100%);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #1b5e20;
    }

    /* ---------- BUTTONS ---------- */
    .stButton > button {
        background: linear-gradient(135deg, #2e7d32 0%, #43a047 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        transform: translateY(-1px);
    }

    /* ---------- TABS ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e8f5e9;
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        color: #1b5e20;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2e7d32 0%, #43a047 100%);
        color: white !important;
    }

    /* ---------- FOOTER ---------- */
    .footer {
        text-align: center;
        padding: 20px;
        color: #555;
        font-size: 12px;
        margin-top: 30px;
        border-top: 1px solid #c8e6c9;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# ============================================================
# 3. SIDEBAR — FILTER & NAVIGATION
# ============================================================
@st.cache_data
def cached_load_data(file_path):
    return load_data(file_path)

def sidebar():
    st.sidebar.markdown("### 🌿 Dashboard Perkebunan")
    st.sidebar.markdown("---")

    # Logo / Header
    st.sidebar.markdown(
        """
        <div style="text-align:center; padding:10px;">
            <div style="font-size:48px;">🌱</div>
            <h3 style="color:#1b5e20; margin:5px 0;">AgriAnalytics</h3>
            <p style="color:#666; font-size:12px;">Produksi Perkebunan Indonesia 2025</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    return st.session_state.get("data", None)

# ============================================================
# 4. LOAD DATA
# ============================================================
DATA_PATH = "data/Produksi Tanaman Perkebunan Menurut Provinsi dan Jenis Tanaman, 2025.xlsx"

try:
    if "data" not in st.session_state:
        raw_df = cached_load_data(DATA_PATH)
        st.session_state["data"] = raw_df
    df_raw = st.session_state["data"]
    df = clean_data(df_raw)
except Exception as e:
    st.error(f"❌ Gagal memuat data: {e}")
    st.info("Pastikan file Excel berada di folder `data/` dengan nama yang sesuai.")
    st.stop()

# Deteksi kolom (fleksibel)
col_province = "Provinsi"
col_commodity = "Komoditas"
col_production = "Produksi (Ton)"
col_area = "Estimasi Luas (Ha)"
col_productivity = "Produktivitas (Ton/Ha)"

# ============================================================
# 5. SIDEBAR FILTERS
# ============================================================
st.sidebar.markdown("### 🔍 Filter Data")

# Pilih komoditas
commodities = ["Semua"] + sorted(df[col_commodity].dropna().unique().tolist())
selected_commodity = st.sidebar.selectbox("🌾 Komoditas", commodities)

# Pilih provinsi
provinces = ["Semua"] + sorted(df[col_province].dropna().unique().tolist())
selected_province = st.sidebar.selectbox("📍 Provinsi", provinces)

st.sidebar.markdown("---")

# Reset button
if st.sidebar.button("🔄 Reset Filter", use_container_width=True):
    st.session_state["selected_commodity"] = "Semua"
    st.session_state["selected_province"] = "Semua"
    st.rerun()

# Apply filter
df_filtered = df.copy()
if selected_commodity != "Semua":
    df_filtered = df_filtered[df_filtered[col_commodity] == selected_commodity]
if selected_province != "Semua":
    df_filtered = df_filtered[df_filtered[col_province] == selected_province]

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Statistik Filter")
st.sidebar.metric("Total Data", f"{len(df_filtered):,}")
st.sidebar.metric("Provinsi", f"{df_filtered[col_province].nunique()}")
st.sidebar.metric("Komoditas", f"{df_filtered[col_commodity].nunique()}")

# ============================================================
# 6. MAIN HEADER
# ============================================================
st.markdown(
    """
    <div class="main-header">
        <h1>🌿 Dashboard Produksi Tanaman Perkebunan</h1>
        <p>Analisis Interaktif Data Perkebunan Indonesia — Tahun 2024 | UAS Data Science</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 7. KPI CARDS
# ============================================================
total_production = df_filtered[col_production].sum() if col_production else 0
total_provinces = df_filtered[col_province].nunique()
total_commodities = df_filtered[col_commodity].nunique()
avg_production = df_filtered[col_production].mean() if col_production else 0

def format_number(x):
    if x >= 1_000_000:
        return f"{x/1_000_000:.2f} Jt"
    elif x >= 1_000:
        return f"{x/1_000:.2f} Rb"
    return f"{x:,.0f}"

st.markdown('<div class="section-title">📈 Ringkasan Utama</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-icon">🌾</div>
            <div class="kpi-value">{format_number(total_production)}</div>
            <div class="kpi-label">Total Produksi (Ton)</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-icon">📍</div>
            <div class="kpi-value">{total_provinces}</div>
            <div class="kpi-label">Total Provinsi</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-icon">🌱</div>
            <div class="kpi-value">{total_commodities}</div>
            <div class="kpi-label">Jenis Komoditas</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-icon">⚖️</div>
            <div class="kpi-value">{format_number(avg_production)}</div>
            <div class="kpi-label">Rata-rata Produksi</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# 8. TABS — VISUALISASI & ANALISIS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Overview", "📈 Tren & Distribusi", "🔬 Analisis Lanjut", "🏆 Top Provinsi", "🤖 Machine Learning", "💡 Insight"]
)

# ---------- TAB 1: OVERVIEW ----------
with tab1:
    st.markdown('<div class="section-title">Produksi per Komoditas & Provinsi</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        # Bar Chart — Produksi per Komoditas
        prod_by_commodity = (
            df_filtered.groupby(col_commodity)[col_production]
            .sum()
            .reset_index()
            .sort_values(col_production, ascending=False)
        )
        fig_bar = px.bar(
            prod_by_commodity,
            x=col_commodity,
            y=col_production,
            title="📊 Total Produksi per Komoditas",
            color=col_production,
            color_continuous_scale="Greens",
            text_auto=".2s",
        )
        fig_bar.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1b5e20"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        # Pie Chart — Distribusi
        fig_pie = px.pie(
            prod_by_commodity.head(8),
            names=col_commodity,
            values=col_production,
            title="🥧 Distribusi Produksi (Top 8)",
            color_discrete_sequence=px.colors.sequential.Greens_r,
            hole=0.4,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1b5e20"),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Data preview
    st.markdown('<div class="section-title">📋 Preview Data</div>', unsafe_allow_html=True)
    st.dataframe(
        df_filtered.head(20).style.background_gradient(cmap="Greens", subset=[col_production]),
        use_container_width=True,
    )

# ---------- TAB 2: TREND & DISTRIBUTION ----------
with tab2:
    st.markdown('<div class="section-title">Distribusi & Tren Produksi</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        # Line chart — produksi per provinsi
        prod_by_province = (
            df_filtered.groupby(col_province)[col_production]
            .sum()
            .reset_index()
            .sort_values(col_production, ascending=False)
        )
        fig_line = px.line(
            prod_by_province,
            x=col_province,
            y=col_production,
            markers=True,
            title="📈 Produksi per Provinsi",
            color_discrete_sequence=["#2e7d32"],
        )
        fig_line.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1b5e20"),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with c2:
        # Histogram — distribusi produksi
        fig_hist = px.histogram(
            df_filtered,
            x=col_production,
            nbins=30,
            title="📊 Distribusi Produksi",
            color_discrete_sequence=["#43a047"],
        )
        fig_hist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1b5e20"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Boxplot
    st.markdown('<div class="section-title">📦 Boxplot Produksi per Komoditas</div>', unsafe_allow_html=True)
    fig_box = px.box(
        df_filtered,
        x=col_commodity,
        y=col_production,
        color=col_commodity,
        color_discrete_sequence=px.colors.sequential.Greens,
        title="📦 Sebaran Produksi per Komoditas",
    )
    fig_box.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1b5e20"),
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ---------- TAB 3: ANALISIS LANJUT ----------
with tab3:
    st.markdown('<div class="section-title">Scatter Plot & Korelasi</div>', unsafe_allow_html=True)

    if col_area is not None:
        c1, c2 = st.columns(2)
        with c1:
            fig_scatter = px.scatter(
                df_filtered.dropna(subset=[col_area, col_production]),
                x=col_area,
                y=col_production,
                color=col_commodity,
                size=col_production,
                hover_data=[col_province],
                title="🎯 Luas Area vs Produksi",
                color_discrete_sequence=px.colors.sequential.Greens,
            )
            fig_scatter.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#1b5e20"),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with c2:
            # Heatmap korelasi
            numeric_df = df_filtered.select_dtypes(include=[np.number])
            if len(numeric_df.columns) >= 2:
                corr = numeric_df.corr()
                fig_heat = px.imshow(
                    corr,
                    text_auto=".2f",
                    color_continuous_scale="Greens",
                    title="🔥 Heatmap Korelasi",
                    aspect="auto",
                )
                fig_heat.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#1b5e20"),
                )
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("Data numerik kurang dari 2 kolom untuk heatmap.")
    else:
        st.info("Kolom 'Luas' tidak ditemukan. Scatter plot tidak dapat ditampilkan.")

# ---------- TAB 4: TOP PROVINSI ----------
with tab4:
    st.markdown('<div class="section-title">🏆 Ranking Provinsi Penghasil Terbesar</div>', unsafe_allow_html=True)

    top_provinces = (
        df_filtered.groupby(col_province)[col_production]
        .sum()
        .reset_index()
        .sort_values(col_production, ascending=False)
        .head(10)
    )
    top_provinces["Rank"] = range(1, len(top_provinces) + 1)

    fig_rank = px.bar(
        top_provinces,
        x=col_production,
        y=col_province,
        orientation="h",
        color=col_production,
        color_continuous_scale="Greens",
        title="🥇 Top 10 Provinsi Penghasil",
        text_auto=".2s",
    )
    fig_rank.update_layout(
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1b5e20"),
        height=500,
    )
    st.plotly_chart(fig_rank, use_container_width=True)

    st.dataframe(
        top_provinces[["Rank", col_province, col_production]].style.background_gradient(
            cmap="Greens", subset=[col_production]
        ),
        use_container_width=True,
    )

# ---------- TAB 5: MACHINE LEARNING ----------
with tab5:
    st.markdown('<div class="section-title">🤖 Prediksi dengan Machine Learning</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="insight-box">
            Model <b>Random Forest</b> dan <b>Linear Regression</b> digunakan untuk memprediksi
            produksi tanaman perkebunan berdasarkan fitur numerik yang tersedia.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        # Siapkan data untuk ML
        ml_df = df_filtered.select_dtypes(include=[np.number]).dropna()
        if col_production in ml_df.columns and len(ml_df) > 20:
            X = ml_df.drop(columns=[col_production])
            y = ml_df[col_production]

            if X.shape[1] >= 1:
                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("##### 🌳 Random Forest Regressor")
                    rf_model, rf_metrics = train_random_forest(X, y)
                    st.metric("MAE", f"{rf_metrics['mae']:,.2f}")
                    st.metric("RMSE", f"{rf_metrics['rmse']:,.2f}")
                    st.metric("R² Score", f"{rf_metrics['r2']:.4f}")

                    # Feature importance
                    feat_imp = get_feature_importance(rf_model, X.columns)
                    fig_imp = px.bar(
                        feat_imp,
                        x="importance",
                        y="feature",
                        orientation="h",
                        title="🌳 Feature Importance (RF)",
                        color="importance",
                        color_continuous_scale="Greens",
                    )
                    fig_imp.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#1b5e20"),
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

                with c2:
                    st.markdown("##### 📐 Linear Regression")
                    lr_model, lr_metrics = train_linear_regression(X, y)
                    st.metric("MAE", f"{lr_metrics['mae']:,.2f}")
                    st.metric("RMSE", f"{lr_metrics['rmse']:,.2f}")
                    st.metric("R² Score", f"{lr_metrics['r2']:.4f}")

                    # Actual vs Predicted
                    y_pred_lr = lr_model.predict(X)
                    fig_pred = px.scatter(
                        x=y,
                        y=y_pred_lr,
                        labels={"x": "Actual", "y": "Predicted"},
                        title="📊 Actual vs Predicted (LR)",
                        color_discrete_sequence=["#2e7d32"],
                    )
                    fig_pred.add_trace(
                        go.Scatter(
                            x=[y.min(), y.max()],
                            y=[y.min(), y.max()],
                            mode="lines",
                            line=dict(color="red", dash="dash"),
                            name="Perfect Prediction",
                        )
                    )
                    fig_pred.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#1b5e20"),
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)

                # Perbandingan model
                st.markdown("##### 📊 Perbandingan Performa Model")
                comparison = pd.DataFrame(
                    {
                        "Model": ["Random Forest", "Linear Regression"],
                        "MAE": [rf_metrics["mae"], lr_metrics["mae"]],
                        "RMSE": [rf_metrics["rmse"], lr_metrics["rmse"]],
                        "R²": [rf_metrics["r2"], lr_metrics["r2"]],
                    }
                )
                st.dataframe(comparison, use_container_width=True)
            else:
                st.warning("Fitur numerik tidak cukup untuk training model.")
        else:
            st.warning("Data tidak cukup untuk training model ML.")
    except Exception as e:
        st.error(f"❌ Error saat training model: {e}")

# ---------- TAB 6: INSIGHT ----------
with tab6:
    st.markdown('<div class="section-title">💡 Insight Otomatis dari Data</div>', unsafe_allow_html=True)

    try:
        top_commodity = (
            df_filtered.groupby(col_commodity)[col_production]
            .sum()
            .sort_values(ascending=False)
            .index[0]
        )
        top_province = (
            df_filtered.groupby(col_province)[col_production]
            .sum()
            .sort_values(ascending=False)
            .index[0]
        )
        max_prod = df_filtered[col_production].max()
        min_prod = df_filtered[col_production].min()

        st.markdown(
            f"""
            <div class="insight-box">
                🌾 <b>Komoditas Unggulan:</b> <b>{top_commodity}</b> merupakan komoditas dengan total produksi tertinggi.
            </div>
            <div class="insight-box">
                📍 <b>Provinsi Terproduktif:</b> <b>{top_province}</b> menjadi penyumbang produksi terbesar.
            </div>
            <div class="insight-box">
                📊 <b>Rentang Produksi:</b> Produksi berkisar antara <b>{format_number(min_prod)}</b> hingga <b>{format_number(max_prod)}</b> ton.
            </div>
            <div class="insight-box">
                📈 <b>Total Data:</b> Dataset berisi <b>{len(df_filtered):,}</b> observasi dari <b>{total_provinces}</b> provinsi dan <b>{total_commodities}</b> komoditas.
            </div>
            <div class="insight-box">
                💡 <b>Rekomendasi:</b> Pemerintah dapat fokus mengembangkan komoditas <b>{top_commodity}</b> di provinsi <b>{top_province}</b> untuk meningkatkan produktivitas nasional.
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Gagal menghasilkan insight: {e}")

# ============================================================
# 9. DOWNLOAD DATA
# ============================================================
st.markdown("---")
st.markdown('<div class="section-title">📥 Download Hasil Analisis</div>', unsafe_allow_html=True)

col_d1, col_d2, col_d3 = st.columns(3)

with col_d1:
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📊 Download Data Filter (CSV)",
        data=csv,
        file_name="data_perkebunan_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_d2:
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df_filtered.to_excel(writer, sheet_name="Data", index=False)
    st.download_button(
        label="📑 Download Data (Excel)",
        data=excel_buffer.getvalue(),
        file_name="data_perkebunan_filtered.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

with col_d3:
    summary = get_summary_stats(df_filtered, col_production)
    summary_csv = summary.to_csv(index=True).encode("utf-8")
    st.download_button(
        label="📈 Download Ringkasan Statistik",
        data=summary_csv,
        file_name="ringkasan_statistik.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ============================================================
# 10. FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        <p>🌿 <b>Dashboard Produksi Tanaman Perkebunan Indonesia</b> — UAS Data Science 2026</p>
        <p>Dibuat dengan ❤️ menggunakan Streamlit, Plotly, dan Scikit-Learn</p>
    </div>
    """,
    unsafe_allow_html=True,
)
