import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Yıldız haritası
star_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

def tum_verileri_cek():
    ana_url = "http://books.toscrape.com/index.html"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'}
    
    response = requests.get(ana_url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Kategori linklerini daha geniş bir seçici ile bulalım
    kategori_etiketleri = soup.select("ul.nav-list > li > ul > li > a")
    
    tum_kitap_listesi = []
    print(f"Toplam {len(kategori_etiketleri)} kategori bulundu.")

    for etiket in kategori_etiketleri:
        kategori_adi = etiket.text.strip()
        # URL'yi düzeltme
        link = etiket["href"].replace("index.html", "")
        # Eğer link 'catalogue' ile başlamıyorsa başına ekle
        if not link.startswith("catalogue/"):
            link = "catalogue/" + link
            
        base_url = f"http://books.toscrape.com/{link}"
        
        print(f"\n>>> {kategori_adi} işleniyor...")
        
        sayfa = 1
        while True:
            current_url = base_url + ("index.html" if sayfa == 1 else f"page-{sayfa}.html")
            res = requests.get(current_url, headers=headers)
            
            if res.status_code != 200:
                break 
                
            s = BeautifulSoup(res.content, "html.parser")
            books = s.find_all("article", class_="product_pod")
            
            if not books: # Sayfada kitap yoksa dur
                break
                
            for book in books:
                title = book.h3.a["title"]
                price_text = book.find("p", class_="price_color").text
                price = float(price_text.replace('£', ''))
                star_class = book.find("p", class_="star-rating")["class"][1]
                rating = star_map[star_class]
                
                tum_kitap_listesi.append({
                    "Kitap Adı": title,
                    "Kategori": kategori_adi,
                    "Fiyat (£)": price,
                    "Puan": rating
                })
            
            sayfa += 1
            time.sleep(0.05) # Hızı biraz artırdık
            
    return tum_kitap_listesi

# --- ÇALIŞTIRMA ---
veriler = tum_verileri_cek()
df = pd.DataFrame(veriler)

if not df.empty:
    print(f"\nBaşarılı! Toplam {len(df)} kitap çekildi.")
    
    # Kayıt işlemleri
    df.to_excel("tum_kitaplar_listesi.xlsx", index=False)
    
    # Ortalama Fiyat Tablosu
    fiyat_ozet = df.groupby("Kategori")[["Fiyat (£)"]].mean().sort_values(by="Fiyat (£)", ascending=False)
    fiyat_ozet.to_excel("kategori_bazli_fiyat_analizi.xlsx")
    
    # Puan Analizi Tablosu
    puan_ozet = df.groupby("Kategori")[["Puan"]].mean().sort_values(by="Puan", ascending=False)
    puan_ozet.to_excel("kategori_bazli_puan_analizi.xlsx")
    
    print("Excel dosyaları oluşturuldu.")
else:
    print("\n[!] HATA: Liste hala boş. İnternet bağlantını veya URL yapısını kontrol et.")
                
 