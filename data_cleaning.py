import pandas as pd

# Ham veriyi oku
df = pd.read_excel("tum_kitaplar_listesi.xlsx")

# 1. 'Add a comment', 'Default' ve 'Nonfiction' kategorilerini temizle
silinecekler = ["Add a comment", "Default", "Nonfiction"]
df_clean = df[~df["Kategori"].isin(silinecekler)].copy()

# Temiz veriyi kaydet
df_clean.to_excel("kitaplar_final_temiz.xlsx", index=False)
print(f"Belirsiz kategoriler temizlendi. Yeni veri seti {len(df_clean)} satır.")