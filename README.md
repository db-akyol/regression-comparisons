# Regresyon Modelleri Karşılaştırması

Bu proje, 1.000 konut kaydından oluşan bir veri setinde **ev fiyatı tahmini** problemini ele alır ve altı farklı regresyon algoritmasını aynı veri bölünmesi, aynı ölçeklendirme ve aynı metrik seti üzerinde karşılaştırır. Karşılaştırılan modeller: Linear, Lasso, Ridge, ElasticNet, SVR (Support Vector Regression) ve Decision Tree Regressor. Modellerin çoğu `GridSearchCV` ile 5 katlı çapraz doğrulama kullanılarak ayarlanmış, sonuçlar MAE / MSE / R² üzerinden tek bir tabloda toplanmıştır.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Veri%20İşleme-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Veri Seti

`house_price_regression_dataset.csv` — **1000 satır × 8 sütun**, eksik değer yok (`df.isna().sum()` tüm sütunlarda 0).

| Sütun | Tip | Açıklama | Aralık |
|---|---|---|---|
| `Square_Footage` | int64 | Evin kullanım alanı (ft²) | 503 – 4999 |
| `Num_Bedrooms` | int64 | Yatak odası sayısı | 1 – 5 |
| `Num_Bathrooms` | int64 | Banyo sayısı | 1 – 3 |
| `Year_Built` | int64 | Yapım yılı | 1950 – 2022 |
| `Lot_Size` | float64 | Arsa büyüklüğü | 0.51 – 4.99 |
| `Garage_Size` | int64 | Garaj kapasitesi (araç) | 0 – 2 |
| `Neighborhood_Quality` | int64 | Mahalle kalite puanı | 1 – 10 |
| **`House_Price`** | float64 | **Hedef değişken** — ev fiyatı | 111.626 – 1.108.236 (ort. ≈ 618.861) |

### Korelasyon bulgusu

Notebook'taki `df.corr()` çıktısına göre hedefle en güçlü ilişki **`Square_Footage` → `House_Price` = 0.9912**. Diğer tüm özelliklerin hedefle korelasyonu zayıftır:

| Özellik | House_Price ile korelasyon |
|---|---|
| Square_Footage | **0.9913** |
| Lot_Size | 0.1604 |
| Garage_Size | 0.0521 |
| Year_Built | 0.0520 |
| Num_Bedrooms | 0.0146 |
| Neighborhood_Quality | -0.0078 |
| Num_Bathrooms | -0.0019 |

Bu tek başına projenin sonucunu açıklayan en kritik bulgudur: veri, neredeyse saf bir **doğrusal** yapıya sahiptir. Aşağıdaki sonuç tablosu bunun doğrudan yansımasıdır.

---

## Karşılaştırılan Modeller

| # | Model | Hiperparametre araması |
|---|---|---|
| 1 | **Linear Regression** | `GridSearchCV` — `fit_intercept`, `copy_X`, `positive` (8 aday, 40 fit) |
| 2 | **Lasso (L1)** | `GridSearchCV` — `alpha` = `np.logspace(-4, 2, 20)`, `max_iter` (60 aday, 300 fit) |
| 3 | **Ridge (L2)** | Varsayılan parametreler (grid search yok) |
| 4 | **ElasticNet (L1+L2)** | `GridSearchCV` — `alpha`, `l1_ratio`, `max_iter` (180 aday, 900 fit) |
| 5 | **SVR (Support Vector Regression)** | `GridSearchCV` — `kernel='linear'`, `C=500000`, `epsilon=100`, `tol` (2 aday, 10 fit) |
| 6 | **Decision Tree Regressor** | `GridSearchCV` — `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `criterion` (480 aday, 2400 fit) |

### Seçilen en iyi parametreler (`grid.best_params_`)

```python
# Linear Regression
{'copy_X': True, 'fit_intercept': True, 'positive': True}

# Lasso
{'alpha': 0.00020691380811147902, 'max_iter': 1000}

# SVR
{'C': 500000, 'cache_size': 100, 'degree': 1, 'epsilon': 100,
 'gamma': 'scale', 'kernel': 'linear', 'max_iter': -1,
 'shrinking': True, 'tol': 0.001}

# Decision Tree
{'criterion': 'squared_error', 'max_depth': None, 'max_features': None,
 'min_samples_leaf': 5, 'min_samples_split': 2}
```

---

## Yöntem

1. **Keşifsel analiz (EDA)** — `df.shape`, `df.head()`, `df.info()`, `df.isna().sum()` ile veri bütünlüğü doğrulandı; fiyat histogramı, `Square_Footage` vs `House_Price` saçılım grafiği ve korelasyon ısı haritası çizildi.
2. **Özellik / hedef ayrımı** — `X = df.drop("House_Price", axis=1)`, `y = df["House_Price"]` (7 özellik).
3. **Train / test ayrımı** — `train_test_split(X, y, test_size=0.25, random_state=15)` → **750 eğitim / 250 test** örneği. `random_state` sabitlendiği için sonuçlar tekrar üretilebilir ve tüm modeller **tamamen aynı bölünme** üzerinde değerlendirilir.
4. **Ölçeklendirme** — `StandardScaler` yalnızca eğitim setine `fit_transform`, test setine `transform` uygulandı. Böylece test verisinden eğitim tarafına bilgi sızması (data leakage) engellendi. Bu adım özellikle düzenlileştirilmiş modeller (Lasso, Ridge, ElasticNet) ve SVR için zorunludur.
5. **Hiperparametre ayarı** — `GridSearchCV(cv=5, scoring='r2', refit=True, n_jobs=-1)`; en iyi model otomatik olarak yeniden eğitilip test setinde tahmin üretti.
6. **Değerlendirme metrikleri** — `mean_absolute_error` (MAE), `mean_squared_error` (MSE), `r2_score` (R²). Her model için gerçek/tahmin saçılım grafiği de çizildi.

---

## Sonuçlar

Tüm skorlar **aynı 250 örneklik test setinde** ölçülmüştür. Tablo MSE'ye göre (küçükten büyüğe) sıralanmıştır.

| Sıra | Model | MAE ↓ | MSE ↓ | RMSE ↓ * | R² ↑ |
|---|---|---|---|---|---|
| 🥇 1 | **Lasso** | 7.389,05 | 87.055.496,37 | 9.330,35 | **0,9985536** |
| 🥈 2 | **Linear Regression** | 7.389,05 | 87.055.497,18 | 9.330,35 | 0,9985536 |
| 🥉 3 | **ElasticNet** | 7.389,59 | 87.067.023,84 | 9.330,97 | 0,9985534 |
| 4 | **Ridge** | 7.398,30 | 87.262.844,54 | 9.341,46 | 0,9985012 † |
| 5 | **SVR (linear)** | **7.363,42** | 87.471.136,78 | 9.352,60 | 0,9985467 |
| 6 | **Decision Tree** | 25.981,27 | 1.031.096.851,09 | 32.110,70 | 0,9828682 |

<sub>\* RMSE notebook'ta ayrıca hesaplanmamıştır; tabloya kolay yorumlanabilirlik için MSE'nin karekökü olarak türetilmiştir (√MSE).</sub>
<sub>† Ridge'in ham R² çıktısı `0.998550116760977`'dir.</sub>

Ek olarak, hiperparametre aramasının katkısı notebook içinde varsayılan (tune edilmemiş) skorlarla birlikte kayıt altına alınmıştır:

| Model | Varsayılan R² | GridSearchCV sonrası R² | Kazanım |
|---|---|---|---|
| ElasticNet | 0,8869595 | **0,9985534** | **+0,1116** (MAE 69.371 → 7.390) |
| Decision Tree | 0,9814246 | **0,9828682** | +0,0014 (MAE 26.447 → 25.981) |
| Linear Regression | 0,9985536 | 0,9985536 | ≈ 0 (zaten doygun) |
| Lasso | 0,9985536 | 0,9985536 | ≈ 0 (zaten doygun) |

### Yorum — hangi model neden kazandı?

- **Doğrusal modeller açık ara kazanıyor.** `Square_Footage` ile `House_Price` arasındaki 0,9912'lik korelasyon, hedefin neredeyse tek bir özelliğin doğrusal fonksiyonu olduğunu gösteriyor. Böyle bir veri üretim sürecinde doğrusal hipotez sınıfı zaten "doğru" model ailesidir; bu yüzden Linear, Lasso, ElasticNet ve SVR(linear) dördü de **R² ≈ 0,9985** bandında, birbirinden yalnızca ondalık basamaklarda ayrılarak kümeleniyor.
- **Lasso birinci, ama farkı istatistiksel gürültü düzeyinde.** Lasso ile Linear arasındaki MSE farkı **0,81** birimdir (87.055.496,37'ye karşı 87.055.497,18) — yani %0,000001'lik bir fark. Lasso'nun seçtiği `alpha ≈ 0,000207` pratikte düzenlileştirmeyi neredeyse kapatıyor, bu da modeli sıradan en küçük karelere yakınsatıyor. **Pratik sonuç: bu veri setinde Lasso "kazanmıyor", doğrusal model ailesi kazanıyor.** Üretimde tercih edilecek model, en yalın ve en hızlı olan **Linear Regression** olmalıdır (Occam'ın usturası).
- **MAE'ye göre lider SVR.** SVR en düşük MAE'yi (7.363,42) üretirken MSE'si en yüksek ikinci sıradadır. Bunun nedeni `epsilon=100` ile tanımlanan epsilon-duyarsız kayıp fonksiyonudur: küçük hataları cezalandırmaz, dolayısıyla tipik (ortanca) hatayı düşürür; buna karşılık birkaç büyük sapmayı toleranse ettiği için karesel hata metriğinde geriye düşer. **Metrik seçimi model seçimini değiştirir** — bu tablonun en öğretici noktası budur.
- **Ridge neden son sıradaki doğrusal model?** Tek grid search yapılmayan model Ridge'tir; varsayılan `alpha=1.0` ile çalışır. Bu, dominant `Square_Footage` katsayısını gereğinden fazla büzerek küçük ama ölçülebilir bir kayıp yaratır (MAE 7.398,30 — doğrusal modeller arasındaki en yüksek değer).
- **Decision Tree neden geride kaldı?** Karar ağacı parçalı sabit (piecewise-constant) tahmin üretir; sürekli ve doğrusal bir hedefi merdiven basamaklarıyla yaklaşık olarak modellemek zorundadır. Sonuç: MAE doğrusal modellerin **~3,5 katı**, MSE ise **~12 katı**. 2.400 fit'lik kapsamlı bir grid search bile bu yapısal dezavantajı kapatamamıştır — **hiperparametre ayarı, yanlış model ailesini kurtarmaz.**
- **Hata büyüklüğü bağlamı:** Ortalama ev fiyatı ≈ 618.861'dir. Doğrusal modellerin MAE'si bunun **yaklaşık %1,2'sine**, Decision Tree'nin MAE'si ise **yaklaşık %4,2'sine** karşılık gelir.

---

## Kurulum ve Çalıştırma

### Gereksinimler

- Python 3.13 (notebook `Python 3 (ipykernel)` / 3.13.5 ile çalıştırılmıştır)
- pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter

### Adımlar

```bash
# 1. Depoyu klonlayın
git clone https://github.com/<kullanici-adi>/regression-comparisons.git
cd regression-comparisons

# 2. Sanal ortam oluşturun
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Bağımlılıkları kurun
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 4. Notebook'u açın
jupyter notebook regression-comparisons.ipynb
```

Hücreleri yukarıdan aşağıya sırayla çalıştırın. `random_state=15` sabit olduğu için yukarıdaki tabloda yer alan skorlar birebir yeniden üretilir.

> **Not:** Decision Tree hücresi 480 aday × 5 kat = **2.400 fit** çalıştırır; ElasticNet hücresi ise 900 fit. `n_jobs=-1` ile tüm CPU çekirdekleri kullanılır, yine de bu iki hücre diğerlerinden belirgin biçimde uzun sürer.

---

## Dosya Yapısı

```
regression-comparisons/
├── regression-comparisons.ipynb        # Ana analiz: EDA, 6 model, GridSearchCV, karşılaştırma
├── house_price_regression_dataset.csv  # Veri seti (1000 satır × 8 sütun)
├── .gitignore                          # Jupyter checkpoint, __pycache__, venv, editör dosyaları
├── LICENSE                             # MIT
└── README.md
```

### Notebook akışı

| Hücre aralığı | İçerik |
|---|---|
| 0 – 2 | Kütüphane importları |
| 3 – 6 | Veri yükleme, `shape`, `head()`, korelasyon matrisi |
| 7 – 9 | Görselleştirme: fiyat histogramı, saçılım grafiği, korelasyon ısı haritası |
| 10 – 15 | Veri kalitesi kontrolü (`isna`, `info`, `columns`) |
| 16 – 20 | X/y ayrımı, train/test bölünmesi, `StandardScaler` |
| 21 – 26 | Linear, Lasso, Ridge, ElasticNet + grid search sonuçları |
| 27 – 30 | SVR + grid search sonuçları |
| 31 – 32 | Decision Tree Regressor + grid search sonuçları |
| 33 – 37 | Metriklerin toplanması, sıralanması ve nihai karşılaştırma çıktısı |

---

## Lisans

Bu proje **MIT Lisansı** ile yayımlanmıştır. Ayrıntılar için [LICENSE](LICENSE) dosyasına bakınız.

Copyright (c) 2025 Deniz Akyol
