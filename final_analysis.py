import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def make_visualizations(df, price_col, rating_col, out_dir='plots'):
	Path(out_dir).mkdir(exist_ok=True)

	# 1) Rating-Category distribution heatmap (counts)
	pivot = df.pivot_table(index='Kategori', columns=rating_col, values=price_col, aggfunc='count', fill_value=0)
	# sort by total count and keep top 20 for readability
	pivot['total'] = pivot.sum(axis=1)
	pivot = pivot.sort_values('total', ascending=False)
	pivot_top = pivot.drop(columns=['total']).head(20)
	pivot_top.to_excel(Path(out_dir) / 'kategori_puan_dagilimi.xlsx')

	plt.figure(figsize=(10,8))
	sns.heatmap(pivot_top, annot=True, fmt='g', cmap='Blues')
	plt.title('Kategori x Puan Dağılımı (Top 20 Kategori)')
	plt.ylabel('Kategori')
	plt.xlabel('Puan (Yıldız)')
	plt.tight_layout()
	heatmap_path = Path(out_dir) / 'yildiz_kategori_dagilimi.png'
	plt.savefig(heatmap_path, dpi=150)
	plt.close()

	# 2) Category total price (sum) bar chart
	cat_total = df.groupby('Kategori')[price_col].sum().sort_values(ascending=False)
	cat_total.to_excel(Path(out_dir) / 'kategori_fiyat_toplam.xlsx')

	top_n = 20
	cat_total_top = cat_total.head(top_n)
	plt.figure(figsize=(10,8))
	sns.barplot(x=cat_total_top.values, y=cat_total_top.index, palette='viridis')
	plt.title(f'Kategori Bazlı Toplam Fiyat (Top {top_n})')
	plt.xlabel('Toplam Fiyat')
	plt.ylabel('Kategori')
	plt.tight_layout()
	bar_path = Path(out_dir) / 'kategori_fiyat_toplam.png'
	plt.savefig(bar_path, dpi=150)
	plt.close()

	print(f"Saved visualizations: {heatmap_path}, {bar_path} and tables in {out_dir}/")


def create_pivot_tables(df, price_col, rating_col, out_dir='plots'):
	"""Create two pilot tables as requested"""
	Path(out_dir).mkdir(exist_ok=True)
	
	# PILOT TABLE 1: Kategoriler (X) × Puanlar (Y)
	# Shows count of books per category-rating combination
	pivot_kategori_puan = df.pivot_table(
		index='Kategori',
		columns=rating_col,
		values=price_col,
		aggfunc='count',
		fill_value=0
	).astype(int)
	
	# Add total column for reference
	pivot_kategori_puan['Toplam'] = pivot_kategori_puan.sum(axis=1)
	pivot_kategori_puan = pivot_kategori_puan.sort_values('Toplam', ascending=False)
	
	# Save to Excel
	pilot1_path = Path(out_dir) / 'PILOT_kategori_puan_tablosu.xlsx'
	pivot_kategori_puan.to_excel(pilot1_path)
	print(f"✓ Pilot Tablo 1 kaydedildi: {pilot1_path}")
	
	# VISUAL 1: Heatmap for Kategori × Puan (top 20 categories)
	pivot_top20 = pivot_kategori_puan.drop(columns=['Toplam']).head(20)
	plt.figure(figsize=(12, 10))
	sns.heatmap(pivot_top20, annot=True, fmt='g', cmap='YlOrRd', cbar_kws={'label': 'Kitap Sayısı'})
	plt.title('Kategori × Puan Dağılımı (Top 20 Kategori)', fontsize=14, fontweight='bold')
	plt.xlabel('Puan (Yıldız)', fontsize=12)
	plt.ylabel('Kategori', fontsize=12)
	plt.tight_layout()
	visual1_path = Path(out_dir) / 'VISUAL_kategori_puan_heatmap.png'
	plt.savefig(visual1_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f"✓ Görsel 1 kaydedildi: {visual1_path}")
	
	# PILOT TABLE 2: Kategoriler (X) × Fiyat Toplamı (Y)
	# Shows total price sum per category
	kategori_fiyat_toplam = df.groupby('Kategori')[price_col].sum().sort_values(ascending=False)
	kategori_fiyat_toplam_df = pd.DataFrame({
		'Kategori': kategori_fiyat_toplam.index,
		'Toplam Fiyat': kategori_fiyat_toplam.values
	})
	
	# Save to Excel
	pilot2_path = Path(out_dir) / 'PILOT_kategori_fiyat_toplam_tablosu.xlsx'
	kategori_fiyat_toplam_df.to_excel(pilot2_path, index=False)
	print(f"✓ Pilot Tablo 2 kaydedildi: {pilot2_path}")
	
	# VISUAL 2: Bar chart for Kategori × Toplam Fiyat (top 20)
	top20_fiyat = kategori_fiyat_toplam.head(20)
	plt.figure(figsize=(12, 10))
	colors = sns.color_palette('coolwarm', n_colors=len(top20_fiyat))
	plt.barh(range(len(top20_fiyat)), top20_fiyat.values, color=colors)
	plt.yticks(range(len(top20_fiyat)), top20_fiyat.index)
	plt.xlabel('Toplam Fiyat (TL)', fontsize=12)
	plt.ylabel('Kategori', fontsize=12)
	plt.title('Kategori Bazlı Toplam Fiyat (Top 20)', fontsize=14, fontweight='bold')
	plt.gca().invert_yaxis()
	plt.tight_layout()
	visual2_path = Path(out_dir) / 'VISUAL_kategori_fiyat_bar.png'
	plt.savefig(visual2_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f"✓ Görsel 2 kaydedildi: {visual2_path}")
	
	return pivot_kategori_puan, kategori_fiyat_toplam_df


def create_price_segmentation_analysis(df, price_col, out_dir='plots'):
	"""Fiyat Segmentasyon Analizi - Pasta Grafiği"""
	Path(out_dir).mkdir(exist_ok=True)
	
	# Remove missing prices
	df_price = df[df[price_col].notna()].copy()
	
	# Define price segments based on percentiles
	q33 = df_price[price_col].quantile(0.33)
	q67 = df_price[price_col].quantile(0.67)
	
	# Categorize
	def categorize_price(price):
		if price <= q33:
			return 'Ucuz'
		elif price <= q67:
			return 'Orta'
		else:
			return 'Pahalı'
	
	df_price['Segment'] = df_price[price_col].apply(categorize_price)
	
	# Count by segment
	segment_counts = df_price['Segment'].value_counts()
	segment_pct = (segment_counts / len(df_price) * 100).round(1)
	
	# Create pie chart
	fig, ax = plt.subplots(figsize=(10, 8))
	colors = ['#70AD47', '#FFC000', '#C55A11']  # Green, Yellow, Orange
	explode = (0.05, 0.05, 0.05)
	
	wedges, texts, autotexts = ax.pie(
		segment_counts, 
		labels=[f'{seg}\n({segment_pct[seg]}%)' for seg in segment_counts.index],
		autopct='%d kitap',
		startangle=90,
		colors=colors,
		explode=explode,
		textprops={'fontsize': 11, 'weight': 'bold'}
	)
	
	# Style the percentage text
	for autotext in autotexts:
		autotext.set_color('white')
		autotext.set_fontsize(10)
	
	ax.set_title(
		f'Fiyat Segmentasyon Analizi\n'
		f'Ucuz: ≤{q33:.2f}₺ | Orta: {q33:.2f}₺-{q67:.2f}₺ | Pahalı: >{q67:.2f}₺',
		fontsize=14,
		fontweight='bold',
		pad=20
	)
	
	plt.tight_layout()
	seg_path = Path(out_dir) / 'ANALIZ_fiyat_segmentasyon.png'
	plt.savefig(seg_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f"✓ Fiyat Segmentasyon Analizi kaydedildi: {seg_path}")
	
	# Save summary table
	segment_summary = pd.DataFrame({
		'Segment': segment_counts.index,
		'Kitap Sayısı': segment_counts.values,
		'Yüzde (%)': segment_pct.values,
		'Fiyat Aralığı': [
			f'≤{q33:.2f}₺',
			f'{q33:.2f}₺ - {q67:.2f}₺',
			f'>{q67:.2f}₺'
		]
	})
	segment_summary.to_excel(Path(out_dir) / 'ANALIZ_fiyat_segmentasyon_tablo.xlsx', index=False)
	
	return segment_summary


def create_pareto_analysis(df, price_col, out_dir='plots'):
	"""Pareto Analizi (80/20 Kuralı) - Envanter Yönetimi"""
	Path(out_dir).mkdir(exist_ok=True)
	
	# Calculate total price (inventory value) by category
	cat_inventory = df.groupby('Kategori')[price_col].sum().sort_values(ascending=False)
	
	# Calculate cumulative percentage
	total_inventory = cat_inventory.sum()
	cat_inventory_pct = (cat_inventory / total_inventory * 100)
	cat_inventory_cumsum = cat_inventory_pct.cumsum()
	
	# Find how many categories make up 80%
	n_categories_80 = (cat_inventory_cumsum <= 80).sum()
	pct_categories_80 = (n_categories_80 / len(cat_inventory) * 100)
	
	# Create Pareto chart
	fig, ax1 = plt.subplots(figsize=(14, 8))
	
	# Show top 20 categories for readability
	top_n = min(20, len(cat_inventory))
	cat_inv_top = cat_inventory.head(top_n)
	cat_cumsum_top = cat_inventory_cumsum.head(top_n)
	
	# Bar chart (left y-axis)
	x_pos = np.arange(len(cat_inv_top))
	bars = ax1.bar(x_pos, cat_inv_top.values, color='#4472C4', alpha=0.7, label='Envanter Değeri')
	ax1.set_xlabel('Kategori', fontsize=12, fontweight='bold')
	ax1.set_ylabel('Toplam Envanter Değeri (₺)', fontsize=12, fontweight='bold', color='#4472C4')
	ax1.tick_params(axis='y', labelcolor='#4472C4')
	ax1.set_xticks(x_pos)
	ax1.set_xticklabels(cat_inv_top.index, rotation=45, ha='right')
	
	# Cumulative line (right y-axis)
	ax2 = ax1.twinx()
	line = ax2.plot(x_pos, cat_cumsum_top.values, color='#C55A11', marker='o', 
					linewidth=2.5, markersize=6, label='Kümülatif %')
	ax2.set_ylabel('Kümülatif Yüzde (%)', fontsize=12, fontweight='bold', color='#C55A11')
	ax2.tick_params(axis='y', labelcolor='#C55A11')
	ax2.set_ylim([0, 105])
	
	# Add 80% reference line
	ax2.axhline(y=80, color='red', linestyle='--', linewidth=2, alpha=0.7, label='80% Hedef')
	ax2.text(len(cat_inv_top)-1, 82, '80% Çizgisi', color='red', fontsize=10, fontweight='bold')
	
	# Title with key insight
	plt.title(
		f'Pareto Analizi: Envanter Yönetimi (80/20 Kuralı)\n'
		f'En değerli {n_categories_80} kategori (toplam kategorilerin %{pct_categories_80:.1f}\'i) '
		f'envaterin %80\'ini oluşturuyor',
		fontsize=13,
		fontweight='bold',
		pad=20
	)
	
	# Combine legends
	lines1, labels1 = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
	
	plt.tight_layout()
	pareto_path = Path(out_dir) / 'ANALIZ_pareto_80_20.png'
	plt.savefig(pareto_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f"✓ Pareto Analizi kaydedildi: {pareto_path}")
	
	# Save detailed Pareto table
	pareto_table = pd.DataFrame({
		'Kategori': cat_inventory.index,
		'Envanter Değeri (₺)': cat_inventory.values.round(2),
		'Yüzde (%)': cat_inventory_pct.values.round(2),
		'Kümülatif (%)': cat_inventory_cumsum.values.round(2)
	})
	pareto_table['ABC Sınıfı'] = pareto_table['Kümülatif (%)'].apply(
		lambda x: 'A (Kritik)' if x <= 80 else ('B (Önemli)' if x <= 95 else 'C (Normal)')
	)
	pareto_table.to_excel(Path(out_dir) / 'ANALIZ_pareto_detay_tablo.xlsx', index=False)
	
	return pareto_table


def create_summary_table_visuals(top_rated, top5_expensive, bottom5_cheap, top5_cats, out_dir='plots'):
	"""Create visual tables from summary statistics"""
	Path(out_dir).mkdir(exist_ok=True)
	
	# TABLE 1: Top Rated Categories
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.axis('tight')
	ax.axis('off')
	
	top_rated_display = top_rated.reset_index()[['Kategori', 'n', 'mean', 'median', 'std']]
	top_rated_display.columns = ['Kategori', 'Kitap Sayısı', 'Ort. Puan', 'Medyan Puan', 'Std. Sapma']
	top_rated_display['Ort. Puan'] = top_rated_display['Ort. Puan'].round(2)
	top_rated_display['Medyan Puan'] = top_rated_display['Medyan Puan'].round(1)
	top_rated_display['Std. Sapma'] = top_rated_display['Std. Sapma'].round(2)
	
	table1 = ax.table(cellText=top_rated_display.values,
					 colLabels=top_rated_display.columns,
					 cellLoc='left',
					 loc='center',
					 colWidths=[0.3, 0.15, 0.15, 0.2, 0.2])
	
	table1.auto_set_font_size(False)
	table1.set_fontsize(10)
	table1.scale(1, 2)
	
	# Style header
	for i in range(len(top_rated_display.columns)):
		table1[(0, i)].set_facecolor('#4472C4')
		table1[(0, i)].set_text_props(weight='bold', color='white')
	
	# Alternate row colors
	for i in range(1, len(top_rated_display) + 1):
		for j in range(len(top_rated_display.columns)):
			if i % 2 == 0:
				table1[(i, j)].set_facecolor('#E7E6E6')
	
	plt.title('En Yüksek Puanlı Kategoriler (Min. 5 Kitap)', fontsize=14, fontweight='bold', pad=20)
	table1_path = Path(out_dir) / 'TABLO_en_yuksek_puanli_kategoriler.png'
	plt.savefig(table1_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f"✓ Tablo 1 kaydedildi: {table1_path}")
	
	# TABLE 2: Top 5 Most Expensive Categories
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.axis('tight')
	ax.axis('off')
	
	top5_exp_display = top5_expensive.reset_index()[['Kategori', 'n', 'mean', 'median', 'std']]
	top5_exp_display.columns = ['Kategori', 'Kitap Sayısı', 'Ort. Fiyat', 'Medyan Fiyat', 'Std. Sapma']
	top5_exp_display['Ort. Fiyat'] = top5_exp_display['Ort. Fiyat'].round(2)
	top5_exp_display['Medyan Fiyat'] = top5_exp_display['Medyan Fiyat'].round(2)
	top5_exp_display['Std. Sapma'] = top5_exp_display['Std. Sapma'].round(2)
	
	table2 = ax.table(cellText=top5_exp_display.values,
					 colLabels=top5_exp_display.columns,
					 cellLoc='left',
					 loc='center',
					 colWidths=[0.3, 0.15, 0.15, 0.2, 0.2])
	
	table2.auto_set_font_size(False)
	table2.set_fontsize(10)
	table2.scale(1, 2)
	
	for i in range(len(top5_exp_display.columns)):
		table2[(0, i)].set_facecolor('#C55A11')
		table2[(0, i)].set_text_props(weight='bold', color='white')
	
	for i in range(1, len(top5_exp_display) + 1):
		for j in range(len(top5_exp_display.columns)):
			if i % 2 == 0:
				table2[(i, j)].set_facecolor('#FCE4D6')
	
	plt.title('En Pahalı 5 Kategori (Ortalama Fiyata Göre)', fontsize=14, fontweight='bold', pad=20)
	table2_path = Path(out_dir) / 'TABLO_en_pahali_kategoriler.png'
	plt.savefig(table2_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f"✓ Tablo 2 kaydedildi: {table2_path}")
	
	# TABLE 3: Top 5 Categories by Count
	fig, ax = plt.subplots(figsize=(8, 4))
	ax.axis('tight')
	ax.axis('off')
	
	top5_cats_df = pd.DataFrame({
		'Kategori': top5_cats.index,
		'Kitap Sayısı': top5_cats.values
	})
	
	table3 = ax.table(cellText=top5_cats_df.values,
					 colLabels=top5_cats_df.columns,
					 cellLoc='left',
					 loc='center',
					 colWidths=[0.6, 0.4])
	
	table3.auto_set_font_size(False)
	table3.set_fontsize(10)
	table3.scale(1, 2)
	
	for i in range(len(top5_cats_df.columns)):
		table3[(0, i)].set_facecolor('#70AD47')
		table3[(0, i)].set_text_props(weight='bold', color='white')
	
	for i in range(1, len(top5_cats_df) + 1):
		for j in range(len(top5_cats_df.columns)):
			if i % 2 == 0:
				table3[(i, j)].set_facecolor('#E2EFDA')
	
	plt.title('En Çok Kitap İçeren 5 Kategori', fontsize=14, fontweight='bold', pad=20)
	table3_path = Path(out_dir) / 'TABLO_en_cok_kitap_kategoriler.png'
	plt.savefig(table3_path, dpi=150, bbox_inches='tight')
	plt.close()
	print(f"✓ Tablo 3 kaydedildi: {table3_path}")


def analyze(path='tum_kitaplar_listesi.xlsx', save_price='kategori_bazli_fiyat_analizi.xlsx', save_rating='kategori_bazli_puan_analizi.xlsx'):
	df = pd.read_excel(path)
	# remove ambiguous categories
	remove = ["Add a comment", "Default", "Nonfiction"]
	df_clean = df[~df['Kategori'].isin(remove)].copy()

	# normalize columns
	df_clean.columns = [c.strip() for c in df_clean.columns]
	price_col = [c for c in df_clean.columns if 'Fiyat' in c][0]
	rating_col = [c for c in df_clean.columns if 'Puan' in c][0]
	df_clean[price_col] = pd.to_numeric(df_clean[price_col], errors='coerce')
	df_clean[rating_col] = pd.to_numeric(df_clean[rating_col], errors='coerce')

	n_total = len(df_clean)
	num_categories = df_clean['Kategori'].nunique()

	cat_counts = df_clean['Kategori'].value_counts()
	top5_cats = cat_counts.head(5)
	prop_top5 = top5_cats.sum() / n_total

	price = df_clean[price_col].dropna()
	price_stats = {
		'mean': float(price.mean()),
		'median': float(price.median()),
		'std': float(price.std()),
		'skew': float(price.skew()),
		'top_1pct': float(price.quantile(0.99))
	}

	cat_price = df_clean.groupby('Kategori')[price_col].agg(['count','mean','median','std']).rename(columns={'count':'n'})
	cat_price['cv'] = cat_price['std'] / cat_price['mean']
	cat_price = cat_price.sort_values('mean', ascending=False)

	overall_mean = price.mean()
	between_var = (cat_price['n'] * (cat_price['mean'] - overall_mean)**2).sum() / cat_price['n'].sum()
	within_var = (cat_price['n'] * (cat_price['std']**2).fillna(0)).sum() / cat_price['n'].sum()
	variance_ratio = float(between_var / (within_var + 1e-9))

	corr = float(df_clean[[price_col, rating_col]].dropna().corr().iloc[0,1])

	cat_rating = df_clean.groupby('Kategori')[rating_col].agg(['count','mean','median','std']).rename(columns={'count':'n'})
	cat_rating = cat_rating.sort_values('mean', ascending=False)

	min_count = 5
	top_rated = cat_rating[cat_rating['n']>=min_count].head(10)

	top5_expensive = cat_price.head(5)
	bottom5_cheap = cat_price.tail(5)

	# save summaries
	cat_price.to_excel(save_price)
	cat_rating.to_excel(save_rating)

	# CREATE PILOT TABLES (NEW!)
	pivot_kategori_puan, kategori_fiyat_toplam = create_pivot_tables(df_clean, price_col, rating_col)

	# create visualizations and supporting tables
	make_visualizations(df_clean, price_col, rating_col)
	
	# CREATE TABLE VISUALIZATIONS FOR SUMMARY STATS
	create_summary_table_visuals(top_rated, top5_expensive, bottom5_cheap, cat_counts.head(5))
	
	# MIS ANALYSES - Executive Insights
	print("\n=== MIS ANALİZLERİ OLUŞTURULUYOR ===")
	segment_summary = create_price_segmentation_analysis(df_clean, price_col)
	pareto_table = create_pareto_analysis(df_clean, price_col)
	print("=== MIS ANALİZLERİ TAMAMLANDI ===\n")

	# formatted report
	lines = []
	lines.append(f"CLEANED_ROWS: {n_total}")
	lines.append('')
	lines.append('TOP 5 CATEGORIES BY COUNT:')
	lines.extend([f"  - {k}: {v}" for k,v in top5_cats.items()])
	lines.append('')
	lines.append('PRICE STATS:')
	for k,v in price_stats.items():
		lines.append(f"  - {k}: {v}")
	lines.append(f"\nPRICE-RATING CORRELATION: {corr:.3f}")
	lines.append('')
	lines.append('TOP 5 MOST EXPENSIVE CATEGORIES (by mean price):')
	lines.append(top5_expensive[['n','mean','median','std']].to_string())
	lines.append('')
	lines.append('TOP RATED CATEGORIES (min count 5):')
	lines.append(top_rated.to_string())
	lines.append('')
	lines.append('SYNTHESIZED TRENDS:')
	lines.append(f"  - Veri setinde {n_total} kitap ve {num_categories} kategori var. En büyük 5 kategori toplamın %{prop_top5*100:.1f} kadarını oluşturuyor.")
	lines.append(f"  - Fiyatlar: mean={price_stats['mean']:.2f}, median={price_stats['median']:.2f}, skew={price_stats['skew']:.2f}.")
	lines.append(f"  - Fiyat-dengesizlik variance_ratio={variance_ratio:.2f} (between/within).")
	lines.append(f"  - Fiyat ile puan korelasyonu: {corr:.3f}.")
	lines.append(f"  - Örnek pahalı kategoriler: {', '.join(list(top5_expensive.index[:3]))} | Örnek ucuz kategoriler: {', '.join(list(bottom5_cheap.index[-3:]))}")

	report = '\n'.join(lines)
	print(report)
	return {
		'n_total': n_total,
		'num_categories': num_categories,
		'price_stats': price_stats,
		'variance_ratio': variance_ratio,
		'corr': corr,
		'top5_cats': top5_cats.to_dict(),
		'top5_expensive': top5_expensive.reset_index().to_dict(orient='records'),
		'top_rated': top_rated.reset_index().to_dict(orient='records')
	}


if __name__ == '__main__':
	analyze()