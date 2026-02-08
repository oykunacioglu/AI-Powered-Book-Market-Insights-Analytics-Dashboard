import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
import os
from dotenv import load_dotenv
from pathlib import Path

# .env dosyasındaki BOOKSTORE_API_KEY değişkenini sisteme yükler
load_dotenv()

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Bookstore AI Dashboard", layout="wide")

st.title("📚 AI Destekli Kitap Pazarı Analizi")
st.markdown("""
Bu dashboard, bir **MIS projesi** kapsamında toplanan kitap verilerinin 
anlık analizini ve **Yapay Zeka** destekli pazar yorumlarını sunar.
""")

# --- YAN MENÜ (SIDEBAR) & GÜVENLİK ---
st.sidebar.header("⚙️ Ayarlar & Filtreler")

# API Key Yönetimi (.env dosyasından çekilir)
api_key = os.getenv("BOOKSTORE_API_KEY")

if api_key:
    st.sidebar.success("✅ Bookstore API Key Yüklendi")
else:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", help=".env dosyası eksikse manuel girin.")

# --- VERİ YÜKLEME ---
@st.cache_data
def load_data():
    file_name = "tum_kitaplar_listesi.xlsx"
    if not Path(file_name).exists():
        return None, None, None
        
    df = pd.read_excel(file_name)
    
    # Gereksiz kategorileri temizle
    remove_list = ["Add a comment", "Default", "Nonfiction"]
    df = df[~df['Kategori'].isin(remove_list)]
    
    # Sütun isimlerini normalize et (boşlukları temizle)
    df.columns = [c.strip() for c in df.columns]
    
    # Fiyat ve Puan sütunlarını otomatik bul ve sayısal yap
    price_col = [c for c in df.columns if 'Fiyat' in c][0]
    rating_col = [c for c in df.columns if 'Puan' in c][0]
    
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')
    
    return df, price_col, rating_col

df, price_col, rating_col = load_data()

# BUG FIX 1: file_name değişkeni tanımlı değildi
file_name = "tum_kitaplar_listesi.xlsx"

if df is None:
    st.error(f"'{file_name}' dosyası bulunamadı! Lütfen dosyanın proje klasöründe olduğundan emin ol.")
    st.stop()

# --- FİLTRELEME ---
kategoriler = ["Tümü"] + sorted(df['Kategori'].unique().tolist())
secilen_kategori = st.sidebar.selectbox("Kategori Filtrele", kategoriler)

df_filtered = df if secilen_kategori == "Tümü" else df[df['Kategori'] == secilen_kategori]

# --- KPI KARTLARI ---
col1, col2, col3 = st.columns(3)
col1.metric("Toplam Kitap", len(df_filtered))
col2.metric("Ortalama Fiyat", f"£{df_filtered[price_col].mean():.2f}")
col3.metric("Ortalama Puan", f"{df_filtered[rating_col].mean():.1f} ⭐")

st.divider()

# --- GÖRSELLEŞTİRME ---
c_left, c_right = st.columns(2)

with c_left:
    # BUG FIX 2: NaN değerleri temizle
    df_viz = df_filtered[df_filtered[price_col].notna()].copy()
    
    # Fiyat Segmentasyonu (Donut Chart)
    bins = [0, 20, 40, 60, 1000]
    labels = ['Ekonomik (<£20)', 'Standart (£20-£40)', 'Premium (£40-£60)', 'Lüks (>£60)']
    df_viz['Segment'] = pd.cut(df_viz[price_col], bins=bins, labels=labels)
    
    fig_pie = px.pie(df_viz, names='Segment', title="Pazar Fiyat Segmentasyonu", hole=0.5,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_pie, use_container_width=True)

with c_right:
    # BUG FIX 3: NaN değerlerini temizle ve scatter plot için veri hazırla
    df_scatter = df_filtered[[price_col, rating_col, 'Kitap Adı', 'Kategori']].dropna()
    
    # Fiyat vs Puan (Scatter Plot)
    fig_scatter = px.scatter(df_scatter, x=price_col, y=rating_col, size=price_col, 
                             hover_data=['Kitap Adı'], color='Kategori' if secilen_kategori == "Tümü" else None,
                             title="Fiyat ve Puan Korelasyon Analizi")
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- YENİ GRAFİKLER ---
st.divider()
st.subheader("📊 Kategori Bazlı Detaylı Analizler")

graf_left, graf_right = st.columns(2)

with graf_left:
    # Kategori Bazlı Toplam Fiyat (Top 20)
    kategori_fiyat_toplam = df_filtered.groupby('Kategori')[price_col].sum().sort_values(ascending=False).head(20).reset_index()
    kategori_fiyat_toplam.columns = ['Kategori', 'Toplam_Fiyat']
    
    if len(kategori_fiyat_toplam) > 0:
        fig_bar_fiyat = px.bar(
            kategori_fiyat_toplam,
            x='Toplam_Fiyat',
            y='Kategori',
            orientation='h',
            title="Kategori Bazlı Toplam Fiyat (Top 20)",
            labels={'Toplam_Fiyat': 'Toplam Fiyat (£)', 'Kategori': 'Kategori'},
            color='Toplam_Fiyat',
            color_continuous_scale='Viridis'
        )
        fig_bar_fiyat.update_layout(showlegend=False, height=600)
        st.plotly_chart(fig_bar_fiyat, use_container_width=True)
    else:
        st.info("Bu filtre için yeterli veri yok.")

with graf_right:
    # En Yüksek Puanlı Kategoriler (Ortalama Puan - Min 5 kitap)
    kategori_puan = df_filtered.groupby('Kategori')[rating_col].agg(['mean', 'count']).reset_index()
    kategori_puan = kategori_puan[kategori_puan['count'] >= 5].sort_values('mean', ascending=False).head(20)
    
    if len(kategori_puan) > 0:
        fig_bar_puan = px.bar(
            kategori_puan,
            x='mean',
            y='Kategori',
            orientation='h',
            title="En Yüksek Puanlı Kategoriler (Min. 5 Kitap)",
            labels={'mean': 'Ortalama Puan ⭐', 'Kategori': 'Kategori'},
            color='mean',
            color_continuous_scale='RdYlGn',
            hover_data=['count']
        )
        fig_bar_puan.update_layout(showlegend=False, height=600)
        st.plotly_chart(fig_bar_puan, use_container_width=True)
    else:
        st.info("Bu filtre için yeterli veri yok (min. 5 kitap gerekli).")

# --- AI ANALİST BÖLÜMÜ ---
st.divider()
st.header("🤖 AI Stratejik İş Analizi")

if st.button("Pazarı AI ile Yorumla"):
    if not api_key:
        st.warning("Analiz için API Key gerekli.")
    else:
        with st.spinner("AI verileri ve pazar trendlerini inceliyor..."):
            try:
                # OpenAI Client kurulumu
                client = OpenAI(api_key=api_key)
                
                # BUG FIX 4: ozet_stats ve ozet_metin karışıklığı düzeltildi
                # Özet istatistikleri metne çevir
                ozet_stats = df_filtered[[price_col, rating_col]].describe().to_string()
                
                # Kategori dağılımı ekle
                kategori_dagılım = df_filtered['Kategori'].value_counts().head(10).to_string()
                
                # Tam özet metin
                ozet_metin = f"""
TEMEL İSTATİSTİKLER:
{ozet_stats}

EN POPÜLER 10 KATEGORİ:
{kategori_dagılım}

TOPLAM KİTAP: {len(df_filtered)}
ORTALAMA FİYAT: £{df_filtered[price_col].mean():.2f}
ORTALAMA PUAN: {df_filtered[rating_col].mean():.2f}/5
"""
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # En verimli maliyet/performans modeli
                    messages=[
                        {"role": "system", "content": "Sen kıdemli bir MIS ve Pazar Analisti uzmanısın. Türkçe ve profesyonel bir dille cevap ver."},
                        {"role": "user", "content": f"Şu kitap verilerini inceleyerek; genel pazar durumu, karlılık fırsatları ve riskler hakkında 3 maddelik analiz yap:\n{ozet_metin}"}
                    ]
                )
                
                st.success("✅ Analiz Başarıyla Tamamlandı!")
                st.markdown(response.choices[0].message.content)
                
            except Exception as e:
                st.error(f"❌ Analiz sırasında bir hata oluştu: {e}")

st.sidebar.markdown("---")
st.sidebar.caption("Bookstore Analysis Dashboard | MIS 2026")