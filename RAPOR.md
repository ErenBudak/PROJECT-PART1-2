# Düzeltme Raporu

Bu rapor `fix/evaluation-and-tuning` branch'inde yapılan değişiklikleri, her birinin sebebini ve yeniden eğitilen modellerin sonuçlarını içerir.

## Kısa özet

- Kodda, sonuçları geçersiz kılan **8 hata** bulundu ve düzeltildi. En önemlileri: hiperparametre araması hiç çalışmıyordu, aranan parametreler modele ulaşmıyordu ve tahminler yanlış satırlarla karşılaştırılıyordu.
- 1. ve 2. bölümde problem tanımı ve veri hazırlığı **değiştirilmedi**: aynı veri setleri, aynı haftalık/yıllık/günlük ortalamalar, aynı 4 hedef, aynı hiperparametre grid'leri.
- Bütün modeller düzeltilmiş kodla yeniden eğitildi. Sonuçlar bu raporun 2. bölümünde.
- Ardından problem bölge bazında yeniden tanımlandı ve Random Forest bu kurulumda Dataset 1'de test R² 0.82 verdi. Eski ve yeni hedefin farkı ile sonuçlar 3. bölümde.
- Düzeltmelerin amacı modelleri daha başarılı göstermek değil, sonuçları **doğru ve savunulabilir** hale getirmekti.

---

## 1. Yapılan değişiklikler ve sebepleri

### 1.1 Hiperparametre araması hiç çalışmıyordu

**Sorun.** `GridSearchCV.fit(train)` çağrısında `y` verilmiyordu, çünkü hedefler (`futuremag` vb.) DataFrame'in içindeydi ve model sınıfı onları kendisi ayırıyordu. Ama `scoring='neg_mean_squared_error'`, her denemeyi puanlamak için `y`'ye ihtiyaç duyar. `y` olmadığı için her deneme hata verdi ve skor NaN oldu.

**Kanıt.** Eski notebook çıktılarının hepsinde şu uyarı vardı: `One or more of the test scores are non-finite: [nan nan nan ...]`. Kaydedilen "en iyi parametreler" de her modelde grid'deki **ilk** değerlerdi (ör. RF: `n_estimators=50, max_depth=4, min_samples_split=10, max_features=3, shiftnum=2`). Bütün skorlar NaN olunca GridSearch listedeki ilk kombinasyonu döndürür.

**Sonucu.** RF için 3.125, XGBoost için 26.244, MLP için ~2.500 kombinasyon eğitildi ama hiçbiri puanlanamadı. "En iyi model" fiilen rastgele seçilmişti.

**Düzeltme.** `evaluation.py` içine `ml_scorer` eklendi: hedefleri modelin kendisinden alır ve doğru hizalanmış satırlarda skor hesaplar. Ayrıca `error_score='raise'` verildi; artık bir hata olursa sessizce NaN'a dönüşmek yerine eğitim durur.

**Skor ölçütü.** Dört hedefin birimleri çok farklı (magnitude ~1, depth yüzlerce km). Ham MSE'de depth ve longitude baskın çıkardı. Bu yüzden her hedefin MSE'si train'deki varyansına bölünüp ortalanıyor ("normalize MSE"); dört hedef eşit ağırlık alıyor.

*Dosyalar: `evaluation.py`, `RandomForestTests.ipynb`, `XgboostTests.ipynb`, `MLPTests.ipynb`, `gelismisdeneme.ipynb`*

### 1.2 Aranan parametreler modele ulaşmıyordu

**Sorun.** RF, XGBoost ve MLP sınıflarında içteki gerçek model (`RandomForestRegressor` vb.) `__init__` içinde kuruluyordu. GridSearch ise önce nesneyi varsayılan değerlerle oluşturur, sonra denenecek parametreleri `set_params` ile atar. Bu atama sadece dıştaki sınıfın alanlarını değiştiriyor, çoktan kurulmuş olan iç modele dokunmuyordu.

**Kanıt.** Kayıtlı eski modellerde dış sınıf `n_estimators=50, max_depth=4` gösterirken içteki gerçek RF `n_estimators=23, max_depth=12` idi (`__init__`'teki varsayılanlar). XGBoost'ta dış sınıf `learning_rate=0.01` derken iç model `0.1` ile eğitilmişti.

**Sonucu.** 1.1 düzeltilse bile arama anlamsız olurdu: `shiftnum` dışındaki hiçbir hiperparametrenin etkisi yoktu, her deneme aynı varsayılan modeli eğitiyordu.

**Düzeltme.** İç model artık `fit()` içinde, o anki parametrelerle kuruluyor.

*Dosya: `MLProject.ipynb` (üç sınıf)*

### 1.3 Tahminler yanlış satırlarla karşılaştırılıyordu

**Sorun.** Lag özellikleri yüzünden model ilk `shiftnum` satır için tahmin üretemez (LSTM'de ilk `sequence_length` satır). Değerlendirme kodu ise gerçek değerleri baştan alıyordu: `test_actual.iloc[:min_len]`. Yani `t` satırı için yapılan tahmin, `t − shiftnum` satırının gerçek değeriyle karşılaştırılıyordu.

**Kanıt.** Eski kayıtlı modellerle train verisinde bile R² negatif çıkıyordu (ör. XGBoost Dataset 1: −0.98). Hizalama düzeltilince aynı modelin train R²'si 0.99 oldu.

**Sonucu.** Raporlanan bütün MSE, MAE, R², eşik ve konum metrikleri kaydırılmış satırlar üzerindendi. Aynı hata LSTM'in Optuna hedefinde de vardı, yani LSTM hiperparametreleri de yanlış bir ölçüte göre seçilmişti.

**Düzeltme.** Her modele `predict_with_targets` metodu eklendi: (tahmin, gerçek değer, satır index'i) üçlüsünü birlikte döndürür, hizalama modelin kendisinden gelir. Test değerlendirmesinde train'in son satırları "geçmiş" olarak ekleniyor; böylece testin ilk satırları da tahmin edilebiliyor ve hiçbir test satırı kaybolmuyor.

*Dosyalar: `MLProject.ipynb`, `DeepProject.ipynb`, `evaluation.py`*

### 1.4 En güncel gözlem kullanılmıyordu

**Sorun.** Lag'ler `shift(1)…shift(k)` ile kuruluyordu, yani özellikler `t−1, t−2, …` dönemlerine aitti. Hedef ise `t+1` dönemiydi. Aradaki `t` dönemi (en güncel bilgi) hiç kullanılmıyordu; model fiilen **iki adım** ilerisini tahmin ediyordu. LSTM penceresinde de aynı boşluk vardı.

**Düzeltme.** Lag'ler artık `t, t−1, …` (`shift(i − 1)`). LSTM'de hedef, pencerenin son satırının bir sonraki dönemi. `LSTMDemo.ipynb` içindeki pencere fonksiyonu da aynı mantığa çekildi; aksi halde yeni modellerle tutarsız olurdu.

*Dosyalar: `MLProject.ipynb`, `DeepProject.ipynb`, `LSTMDemo.ipynb`*

### 1.5 XGBoost ve RF'de hedef ölçeği

**Sorun (XGBoost).** XGBoost çok çıktılı regresyonda dört hedef için **tek bir** başlangıç tahmini kullanıyor: tüm hedeflerin ortalaması (bu veride ≈ −15.6). Magnitude (~1.6) bu değerden başlıyor ve düşük `learning_rate` / az ağaçla gerçek seviyesine ulaşamıyordu. Bu hata 1.2 düzeltilince ortaya çıktı: deneme eğitiminde magnitude için train R² = −71 çıktı.

**Sorun (RF).** Çok çıktılı ağaç, bölünme kalitesini dört hedefin MSE ortalamasıyla ölçer. Ölçeklenmemiş hedeflerle bu ortalama neredeyse tamamen depth ve longitude tarafından belirleniyordu; magnitude bölünme kararlarını etkilemiyordu.

**Düzeltme.** İki modelde de hedefler eğitimde standartlaştırılıyor (`StandardScaler`), tahminler gerçek birime geri çevriliyor. MLP ve LSTM zaten hedefleri ölçekliyordu.

*Dosya: `MLProject.ipynb`*

### 1.6 Outlier temizliği test verisini görüyordu, RobustScaler etkisizdi

**Sorun.** `gelismisdeneme.ipynb`'de IsolationForest, train/test ayrımından **önce** tüm veriye uygulanıyordu. Yani test setindeki "zor" günler de silinmişti (veri sızıntısı; testi yapay olarak kolaylaştırır). Ayrıca aykırı günler silindiği için zaman çizelgesinde boşluklar oluşuyor, "bir önceki gün" bazen birkaç gün öncesi oluyordu. LSTM+Attention sınıfı da aykırı satırları siliyordu; pencereler ardışık olmayan günlerden oluşuyordu.

RobustScaler ise ölçeklenmiş değerleri `mag_robust` gibi yeni sütunlara yazıyordu, ama modeller özellik olarak hâlâ orijinal `mag, depth, latitude, longitude` sütunlarını kullanıyordu. Yani hiçbir etkisi yoktu.

**Düzeltme.** IsolationForest sadece train'e fit ediliyor; teste dokunulmuyor. Aykırı satırlar silinmiyor, değerleri komşu dönemlerden interpolasyonla dolduruluyor (zaman çizelgesi kesintisiz kalıyor). Etkisiz RobustScaler adımı kaldırıldı. `use_robust_scaling` parametresi, demo notebook'u bozulmasın diye sınıfta duruyor ama bir şey yapmıyor.

*Dosyalar: `gelismisdeneme.ipynb`, `DeepProject.ipynb`*

### 1.7 Overfit görünmüyordu

**Sorun.** Notebook'larda sadece test skoru hesaplanıyordu. Train skoru olmadan bir modelin ezberleyip ezberlemediği anlaşılamaz. Karşılaştırılacak basit bir referans da yoktu; "R² = −0.8 iyi mi kötü mü" sorusunun cevabı yoktu.

**Düzeltme.** Her model için artık yan yana raporlanıyor:
- **Train** ve **test** metrikleri
- **Persistence** referansı: "gelecek dönem = bu dönem"
- **Train ortalaması** referansı: her zaman train'deki ortalamayı tahmin et

Bir model bu iki basit referansı geçemiyorsa, geçmişten anlamlı bir şey öğrenmemiş demektir.

*Dosya: `evaluation.py` (`evaluate_model`)*

### 1.8 Eşik analizi tek sınıflıydı

**Sorun.** Eşikler sabitti (Dataset 1 için 2.0–4.0, Dataset 2 için 6.0–8.0). Ama haftalık/yıllık **ortalama** büyüklük bu değerlere hiç ulaşmıyor. Test setinde "eşik üstü" örnek sayısı 0 olunca model de 0 tahmin ediyor ve accuracy = 1.000 çıkıyordu. Bu sayı hiçbir şey ölçmüyordu.

**Düzeltme.** Eşikler train'deki `futuremag` dağılımından seçiliyor (medyan, %75 ve %90'lık dilim); böylece iki sınıf da mevcut oluyor. Sonuç "çoğunluk sınıfını tahmin et" referansıyla birlikte veriliyor. Testte yine tek sınıf kalırsa metrik hesaplanmıyor ve bu açıkça yazılıyor.

*Dosya: `evaluation.py` (`default_thresholds`, `threshold_report`)*

### 1.9 Diğer değişiklikler

| Değişiklik | Sebep |
|---|---|
| Yeni `evaluation.py` modülü | Aynı değerlendirme kodu notebook'larda 13 kez kopyalanmıştı. Hata her kopyada ayrı ayrı vardı; tek yerde düzeltmek için ortak modüle taşındı. Notebook'lar yaklaşık 2.500 satır kısaldı. |
| Yeni `results/` klasörü | Her modelin train/test/referans metrikleri JSON olarak kaydediliyor. |
| `XgboostTests`: tek `GridSearchCV` yerine `chunked_grid_search` | 26.244 kombinasyonluk tek aramada görevleri işçilere dağıtan ana işlem tıkanıyor, 20 işçi boşta bekliyordu; ilk deney 2,5 saatte bitmedi (tahmini ~8 saat). Aynı kombinasyonlar 250'lik parçalar halinde aranıyor. Aynı adaylar, aynı CV katmanları ve aynı scorer kullanıldığı için seçilen model aynıdır; 2.916 kombinasyonluk bir denemede iki yöntem aynı parametreleri ve aynı skoru verdi, parçalı arama ~2,6 kat hızlıydı. |
| RF/XGBoost içinde `n_jobs=1` | GridSearch zaten tüm çekirdekleri kullanıyor; iç modelin de paralel çalışması çekirdekleri aşırı yüklüyordu. |
| `XgboostTests`'ten `sklearnex` kaldırıldı | Sadece hızlandırma amaçlıydı ve bu ortamda (numpy uyumsuzluğu) import hatası veriyordu. |
| Optuna'da `suggest_loguniform` → `suggest_float(log=True)` | Eski fonksiyon kullanımdan kalkmış. |
| LSTM Optuna hedefi: sadece magnitude MSE → 4 hedefin normalize MSE'si | ML modellerindeki GridSearch ile aynı ölçüt olsun diye. |
| `sLSTMAttentiondataset2.keras.json` → `LSTMAttentiondataset2.keras.json` | Dosya adındaki yazım hatası. |
| `best_model_mlp.json`: `1_week` → `1_day` | O deney günlük veriyle çalışıyor. |
| MLPTests'ten "extra" deneyi çıkarıldı | İstek üzerine (uzun sürdüğü için). |
| `README.md` güncellendi | Düzeltmelerin özeti eklendi; "RobustScaler ile iyileşti" ifadesi kaldırıldı. |
| `.gitignore` eklendi | `__pycache__` klasörü repoya girmesin. |

### 1.10 Değiştirilmeyenler ve bilinen sınırlar

- **Veri miktarı aynı.** Dataset 1 haftalık kurulumda hâlâ ~40 satır, test seti ~9 satır. Bu kadar küçük bir testte R² tek bir satırdan bile çok etkilenir; Dataset 1 haftalık sonuçları istatistiksel olarak zayıftır.
- **Problem tanımı aynı.** Hedefler hâlâ dönem **ortalamaları**. USGS verisi küresel olduğu için enlem/boylam ortalaması gerçek bir konuma karşılık gelmiyor; ortalama büyüklük de büyük depremleri bastırıyor.
- **Hiperparametre grid'leri aynı.** Senin belirlediğin değerler kullanıldı. (XGBoost'ta arama parçalara bölündü ama denenen kombinasyonlar aynı.)
- **ML demo notebook'ları** (`RFDemo`, `XGBoostDemo`, `MLPDemo`) değiştirilmedi ve yeni modellerle yeniden çalıştırılmadı. Sınıfların arayüzü aynı kaldığı için çalışmaları beklenir, ama doğrulanmadı.
- **`models/mlp_extra.pkl`** eski koddan kalma; bu branch'te yeniden eğitilmedi ve yeni kodla tutarlı değil.
- LSTM'de ölçekleyiciler train'in tamamına fit ediliyor; bunun içinde erken durdurma için ayrılan %20'lik validation da var. Test verisi görülmüyor, etkisi küçük; demo notebook'u ile tutarlılık için böyle bırakıldı.

---

## 2. Sonuçlar

### Tablolar nasıl okunur

- **R²:** 1 kusursuz; 0, o veri kümesinin ortalamasını tahmin etmekle aynı; negatif ondan kötü.
- **Train R² yüksek, test R² düşük** → model ezberlemiş (overfit).
- **Train R² ≈ 0** → model neredeyse sabit bir değer (ortalama) tahmin ediyor; ezber yok ama öğrenilen bir şey de yok.
- **Normalize MSE:** dört hedefin tek sayıda özeti; düşük olan iyi. Train ortalaması referansı için tipik değer 1 civarıdır; test dönemi train'den farklıysa daha yüksek çıkar.
- Bir modelin işe yaradığını söylemek için **iki referansı da** (train ortalaması ve persistence) geçmesi gerekir.

Metriklerin oluşturulma zamanı: 2026-10-08 04:34

### Sonuçların yorumu

**Genel sonuç: hiçbir model, dönem ortalamalarındaki değişimi tahmin edemiyor.** 12 modelin tamamında, dört hedefin neredeyse hepsi için test R² negatif. Düzeltmelerden sonra bu artık bir kod hatasının değil, problemin kendisinin sonucu: haftalık/yıllık ortalama büyüklük, derinlik ve konum geçmiş ortalamalardan öngörülemiyor.

**Dataset 1 (USGS 2022, haftalık ve günlük): öğrenilen bir şey yok.**
- RF, XGBoost ve Simple LSTM, "her zaman train ortalamasını tahmin et" referansıyla başa baş; MLP ve LSTM+Attention ondan kötü.
- RF ve LSTM'lerde train R² ≈ 0: hiperparametre araması artık gerçekten çalıştığı için en az ezberleyen ayarları seçiyor ve modeller sabit bir değer tahmin etmeye çöküyor. XGBoost train'de 0.5'e çıkıyor ama testte bunun karşılığı yok (hafif overfit).
- Günlük MLP + IsolationForest özet ölçütte referansların biraz önünde (2.18'e karşı 2.33), ama dört hedefin hepsinde test R² negatif. Bu fark anlamlı bir öğrenme göstermiyor.
- Haftalık kurulumda test seti 9 satır; bu sayılar tek bir haftadan bile çok etkilenir.

**Dataset 2 (1900-2023, yıllık): modeller seviyeyi takip ediyor, yıllık değişimi açıklayamıyor.**
- Bu veride güçlü bir zaman eğilimi var: train ortalaması referansı magnitude için R² = −127 veriyor, yani test dönemindeki (yaklaşık son 24 yıl) yıllık ortalamalar train dönemindekinden çok farklı. Muhtemel sebep kayıt kapsamının zamanla artması (son on yıllarda daha çok küçük deprem kaydediliyor, ortalama düşüyor); bunu ayrıca doğrulamadım.
- RF, XGBoost ve MLP hem train ortalamasını hem persistence'ı geçiyor (normalize MSE 0.34-0.43, persistence 0.46). Ancak test R² yine negatif: modeller son yılların seviyesini yakalıyor, yıldan yıla oynamayı açıklayamıyor. Persistence'a göre kazanç sınırlı.
- **Overfit belirgin:** train R² 0.92-0.999 iken test R² negatif. En uç örnek XGBoost (train 0.999). ~90 satırlık train ile bu beklenen bir sonuç.
- İki LSTM de persistence'ın gerisinde; 80 satır bir LSTM için çok az.

**RF Extra (Dataset 2, haftalık, ~4.100 train satırı):** Yeterli veri olan tek kurulum. Referansları geçiyor (0.72'ye karşı 0.93 ve 1.21) ve train/test farkı daha küçük (train R² 0.2-0.46). Yine de test R² dört hedefte de negatif.

**Eşik analizi:**
- Dataset 1'de modeller ortalamaya yakın tahmin ettiği için tahminler eşiklerin altında kalıyor; sonuçlar "çoğunluk sınıfını tahmin et" referansının gerisinde.
- Dataset 2'de test yıllarının ortalama büyüklüğü, train'in medyan eşiğinin bile altında (yukarıdaki eğilim yüzünden). Testte tek sınıf kalıyor ve metrik hesaplanmıyor. Eskiden burada accuracy = 1.000 raporlanıyordu.

**Eski sonuçlarla karşılaştırma.** Eski notebook'lardaki sayılar kaydırılmış satırlar üzerinden hesaplandığı ve modeller farklı (bir adım boşluklu) bir hedefe göre eğitildiği için doğrudan karşılaştırılamaz. Özellikle eski "en iyi sonuç" (MLP + IsolationForest, magnitude R² = 0.69) düzeltilmiş kurulumda görünmüyor: aynı deneyde magnitude test R² = −0.33.

**Bundan çıkan sonuç.** Sorun model seçiminde ya da hiperparametrelerde değil, problem tanımında: küresel verinin dönem ortalaması büyük depremleri bastırıyor, konumu anlamsızlaştırıyor ve veriyi onlarca satıra indiriyor. Anlamlı bir sonuç için hedefin ve verinin yeniden tanımlanması gerekir (ör. bölge bazlı zaman serileri ve "bir sonraki dönemin en büyük depremi" hedefi).

### Özet: normalize test MSE (4 hedefin ortalaması, düşük = iyi)

Her hedefin MSE'si train'deki varyansına bölünüp ortalanır. `Train ort.` her zaman train ortalamasını tahmin eden referans, `Persistence` ise "gelecek dönem = bu dönem" referansıdır.

| Model | Veri | Train / Test satır | Model | Train ort. | Persistence | Referansları geçti mi? |
|---|---|---|---|---|---|---|
| RF Dataset1 | Dataset 1, haftalık | 27 / 9 | 4.339 | 4.258 | 5.614 | Başa baş |
| XGB Dataset1 | Dataset 1, haftalık | 31 / 9 | 4.335 | 4.290 | 5.608 | Başa baş |
| MLP Dataset1 | Dataset 1, haftalık | 31 / 9 | 4.676 | 4.290 | 5.608 | Hayır |
| MLP Daily IsolationForest | Dataset 1, günlük | 226 / 57 | 2.177 | 2.330 | 2.758 | Evet |
| Simple LSTM Dataset1 | Dataset 1, günlük | 218 / 57 | 1.678 | 1.652 | 1.997 | Başa baş |
| LSTM Attention Dataset1 | Dataset 1, günlük | 219 / 57 | 1.745 | 1.655 | 2.001 | Hayır |
| RF Dataset2 | Dataset 2, yıllık | 92 / 24 | 0.358 | 1.441 | 0.463 | Evet |
| XGB Dataset2 | Dataset 2, yıllık | 91 / 24 | 0.427 | 1.462 | 0.462 | Evet |
| MLP Dataset2 | Dataset 2, yıllık | 92 / 24 | 0.341 | 1.441 | 0.463 | Evet |
| Simple LSTM Dataset2 | Dataset 2, yıllık | 81 / 24 | 0.961 | 1.741 | 0.485 | Hayır |
| LSTM Attention Dataset2 | Dataset 2, yıllık | 80 / 24 | 1.225 | 1.755 | 0.482 | Hayır |
| RF Extra | Dataset 2, haftalık | 4118 / 1155 | 0.721 | 0.934 | 1.214 | Evet |

### Hedef bazında R² (train ve test)

R² = 0, test ortalamasını tahmin etmekle aynı; negatif değer ondan kötü demek.

| Model | Hedef | Train R² | Test R² | Train ort. R² | Persistence R² | Test MAE | Train ort. MAE |
|---|---|---|---|---|---|---|---|
| RF Dataset1 | Magnitude | 0.029 | -2.125 | -1.954 | -1.174 | 0.169 | 0.164 |
| RF Dataset1 | Depth | 0.019 | -0.655 | -0.651 | -1.225 | 3.020 | 3.022 |
| RF Dataset1 | Latitude | 0.024 | -0.185 | -0.148 | -0.710 | 1.164 | 1.125 |
| RF Dataset1 | Longitude | 0.019 | -0.047 | -0.084 | -1.275 | 4.134 | 4.313 |
| XGB Dataset1 | Magnitude | 0.539 | -1.937 | -1.954 | -1.174 | 0.169 | 0.164 |
| XGB Dataset1 | Depth | 0.496 | -0.802 | -0.651 | -1.225 | 3.143 | 3.022 |
| XGB Dataset1 | Latitude | 0.553 | -0.073 | -0.148 | -0.710 | 1.078 | 1.125 |
| XGB Dataset1 | Longitude | 0.518 | -0.048 | -0.084 | -1.275 | 4.312 | 4.313 |
| MLP Dataset1 | Magnitude | 0.055 | -2.860 | -1.954 | -1.174 | 0.181 | 0.164 |
| MLP Dataset1 | Depth | -0.353 | -0.514 | -0.651 | -1.225 | 2.666 | 3.022 |
| MLP Dataset1 | Latitude | -0.238 | 0.033 | -0.148 | -0.710 | 1.122 | 1.125 |
| MLP Dataset1 | Longitude | 0.177 | -0.338 | -0.084 | -1.275 | 4.589 | 4.313 |
| MLP Daily IsolationForest | Magnitude | 0.077 | -0.330 | -0.749 | -0.137 | 0.158 | 0.181 |
| MLP Daily IsolationForest | Depth | 0.104 | -0.085 | -0.192 | -0.706 | 4.126 | 4.120 |
| MLP Daily IsolationForest | Latitude | -0.050 | -0.155 | -0.016 | -0.451 | 1.714 | 1.630 |
| MLP Daily IsolationForest | Longitude | 0.089 | -0.208 | -0.163 | -0.604 | 5.203 | 5.227 |
| Simple LSTM Dataset1 | Magnitude | -0.001 | -0.713 | -0.676 | -0.137 | 0.179 | 0.177 |
| Simple LSTM Dataset1 | Depth | -0.000 | -0.166 | -0.167 | -0.706 | 4.086 | 4.087 |
| Simple LSTM Dataset1 | Latitude | -0.002 | -0.008 | -0.012 | -0.451 | 1.631 | 1.631 |
| Simple LSTM Dataset1 | Longitude | -0.005 | -0.227 | -0.166 | -0.604 | 5.382 | 5.234 |
| LSTM Attention Dataset1 | Magnitude | -0.021 | -0.909 | -0.676 | -0.137 | 0.192 | 0.177 |
| LSTM Attention Dataset1 | Depth | -0.005 | -0.214 | -0.167 | -0.706 | 4.151 | 4.087 |
| LSTM Attention Dataset1 | Latitude | -0.003 | -0.005 | -0.012 | -0.451 | 1.632 | 1.631 |
| LSTM Attention Dataset1 | Longitude | 0.000 | -0.183 | -0.166 | -0.604 | 5.276 | 5.234 |
| RF Dataset2 | Magnitude | 0.935 | -0.136 | -127.413 | -0.674 | 0.016 | 0.232 |
| RF Dataset2 | Depth | 0.943 | -0.341 | -4.647 | -0.614 | 11.683 | 23.533 |
| RF Dataset2 | Latitude | 0.830 | -0.580 | -10.731 | -0.950 | 3.936 | 12.950 |
| RF Dataset2 | Longitude | 0.601 | -0.169 | -0.426 | -0.576 | 12.748 | 13.507 |
| XGB Dataset2 | Magnitude | 0.999 | -0.281 | -127.413 | -0.674 | 0.020 | 0.232 |
| XGB Dataset2 | Depth | 0.999 | -0.675 | -4.647 | -0.614 | 13.434 | 23.533 |
| XGB Dataset2 | Latitude | 0.999 | -0.872 | -10.731 | -0.950 | 4.206 | 12.950 |
| XGB Dataset2 | Longitude | 0.999 | -0.377 | -0.426 | -0.576 | 13.997 | 13.507 |
| MLP Dataset2 | Magnitude | 0.924 | -0.279 | -127.413 | -0.674 | 0.019 | 0.232 |
| MLP Dataset2 | Depth | 0.940 | 0.003 | -4.647 | -0.614 | 8.896 | 23.533 |
| MLP Dataset2 | Latitude | 0.742 | -0.699 | -10.731 | -0.950 | 3.919 | 12.950 |
| MLP Dataset2 | Longitude | 0.187 | -0.139 | -0.426 | -0.576 | 12.443 | 13.507 |
| Simple LSTM Dataset2 | Magnitude | 0.302 | -16.876 | -127.413 | -0.674 | 0.083 | 0.232 |
| Simple LSTM Dataset2 | Depth | -0.059 | -6.340 | -4.647 | -0.614 | 27.425 | 23.533 |
| Simple LSTM Dataset2 | Latitude | -0.219 | -0.253 | -10.731 | -0.950 | 3.332 | 12.950 |
| Simple LSTM Dataset2 | Longitude | -2.272 | -1.462 | -0.426 | -0.576 | 19.433 | 13.507 |
| LSTM Attention Dataset2 | Magnitude | -0.641 | -0.211 | -127.413 | -0.674 | 0.017 | 0.232 |
| LSTM Attention Dataset2 | Depth | -0.393 | -9.825 | -4.647 | -0.614 | 34.198 | 23.533 |
| LSTM Attention Dataset2 | Latitude | -1.491 | -0.723 | -10.731 | -0.950 | 4.233 | 12.950 |
| LSTM Attention Dataset2 | Longitude | -2.582 | -2.120 | -0.426 | -0.576 | 22.113 | 13.507 |
| RF Extra | Magnitude | 0.464 | -0.255 | -2.048 | -0.893 | 0.128 | 0.218 |
| RF Extra | Depth | 0.447 | -0.097 | -0.192 | -0.969 | 40.458 | 37.404 |
| RF Extra | Latitude | 0.292 | -0.178 | -0.794 | -0.793 | 11.499 | 14.857 |
| RF Extra | Longitude | 0.215 | -0.034 | -0.061 | -0.752 | 44.112 | 44.355 |

### Seçilen hiperparametreler

**RandomForestTests.ipynb**

```
En iyi parametreler: {'max_depth': 4, 'max_features': 3, 'min_samples_split': 20, 'n_estimators': 250, 'shiftnum': 6}
En iyi CV skoru (negatif normalize MSE): -1.826
=== RF Dataset1 | train: 27 satır, test: 9 satır ===
En iyi parametreler: {'max_depth': 9, 'max_features': 8, 'min_samples_split': 10, 'n_estimators': 100, 'shiftnum': 4}
En iyi CV skoru (negatif normalize MSE): -5.734
=== RF Dataset2 | train: 92 satır, test: 24 satır ===
En iyi parametreler: {'max_depth': 13, 'max_features': 6, 'min_samples_split': 32, 'n_estimators': 80, 'shiftnum': 2}
En iyi CV skoru (negatif normalize MSE): -1.187
=== RF Extra | train: 4118 satır, test: 1155 satır ===
```

**XgboostTests.ipynb**

```
En iyi parametreler: {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 5, 'min_child_weight': 1, 'n_estimators': 100, 'reg_alpha': 0.5, 'reg_lambda': 2, 'shiftnum': 2, 'subsample': 0.8}
En iyi CV skoru (negatif normalize MSE): -1.859
=== XGB Dataset1 | train: 31 satır, test: 9 satır ===
En iyi parametreler: {'colsample_bytree': 0.9, 'learning_rate': 0.15, 'max_depth': 4, 'min_child_weight': 3, 'n_estimators': 400, 'reg_alpha': 0.3, 'reg_lambda': 1, 'shiftnum': 5, 'subsample': 0.7}
En iyi CV skoru (negatif normalize MSE): -4.405
=== XGB Dataset2 | train: 91 satır, test: 24 satır ===
```

**MLPTests.ipynb**

```
En iyi parametreler: {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (100, 50), 'learning_rate_init': 0.1, 'max_iter': 500, 'shiftnum': 2, 'solver': 'adam'}
En iyi CV skoru (negatif normalize MSE): -1.632
=== MLP Dataset1 | train: 31 satır, test: 9 satır ===
En iyi parametreler: {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': (200, 100), 'learning_rate_init': 0.02, 'max_iter': 900, 'shiftnum': 4, 'solver': 'adam'}
En iyi CV skoru (negatif normalize MSE): -3.880
=== MLP Dataset2 | train: 92 satır, test: 24 satır ===
```

**gelismisdeneme.ipynb**

```
En iyi parametreler: {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (100, 50), 'learning_rate_init': 0.1, 'max_iter': 500, 'shiftnum': 2, 'solver': 'adam'}
En iyi CV skoru (negatif normalize MSE): -1.015
=== MLP Daily IsolationForest | train: 226 satır, test: 57 satır ===
```

**denemeRNN.ipynb**

```
En iyi parametreler: {'sequence_length': 10, 'lstm_units': 32, 'lstm_layers': 1, 'dropout_rate': 0.1633458693132066, 'dense_dim': 128, 'learning_rate': 0.0002910298159209187, 'batch_size': 16, 'optimizer': 'rmsprop', 'activation': 'tanh'}
=== Simple LSTM Dataset1 | train: 218 satır, test: 57 satır ===
En iyi parametreler: {'sequence_length': 9, 'lstm_units': 128, 'lstm_layers': 3, 'dropout_rate': 0.2949122356790004, 'dense_dim': 128, 'learning_rate': 0.00011758095640770696, 'batch_size': 16, 'optimizer': 'rmsprop', 'activation': 'tanh'}
=== LSTM Attention Dataset1 | train: 219 satır, test: 57 satır ===
En iyi parametreler: {'sequence_length': 15, 'lstm_units': 256, 'lstm_layers': 2, 'dropout_rate': 0.422976062065625, 'dense_dim': 16, 'learning_rate': 0.0007148510793512986, 'batch_size': 32, 'optimizer': 'adam', 'activation': 'relu'}
=== Simple LSTM Dataset2 | train: 81 satır, test: 24 satır ===
En iyi parametreler: {'sequence_length': 16, 'lstm_units': 256, 'lstm_layers': 3, 'dropout_rate': 0.21163982588318786, 'dense_dim': 16, 'learning_rate': 0.000316038032592918, 'batch_size': 64, 'optimizer': 'adam', 'activation': 'relu'}
=== LSTM Attention Dataset2 | train: 80 satır, test: 24 satır ===
```

### Eşik analizi (magnitude tahmini eşiğe göre sınıflandırmaya çevrilince)

**RandomForestTests.ipynb**

```
threshold_report(test_frame, default_thresholds(train))
Büyüklük eşiği: 1.621  |  test: 9 örnek, gerçek pozitif: 8, tahmin pozitif: 0
  Accuracy=0.111 (çoğunluk sınıfı referansı: 0.889)
  Precision=0.000, Recall=0.000, F1=0.000 (hep 'büyük' de referansı F1: 0.941)
  Confusion matrix: TN=1, FP=0, FN=8, TP=0

Büyüklük eşiği: 1.661  |  test: 9 örnek, gerçek pozitif: 7, tahmin pozitif: 0
  Accuracy=0.222 (çoğunluk sınıfı referansı: 0.778)
  Precision=0.000, Recall=0.000, F1=0.000 (hep 'büyük' de referansı F1: 0.875)
  Confusion matrix: TN=2, FP=0, FN=7, TP=0

Büyüklük eşiği: 1.696  |  test: 9 örnek, gerçek pozitif: 6, tahmin pozitif: 0
  Accuracy=0.333 (çoğunluk sınıfı referansı: 0.667)
  Precision=0.000, Recall=0.000, F1=0.000 (hep 'büyük' de referansı F1: 0.800)
  Confusion matrix: TN=3, FP=0, FN=6, TP=0

threshold_report(test_frame_2, default_thresholds(train_2))
Büyüklük eşiği: 6.078  |  test: 24 örnek, gerçek pozitif: 0, tahmin pozitif: 0
  Testte tek sınıf var; bu eşikte sınıflandırma metrikleri anlamsız.

Büyüklük eşiği: 6.255  |  test: 24 örnek, gerçek pozitif: 0, tahmin pozitif: 0
  Testte tek sınıf var; bu eşikte sınıflandırma metrikleri anlamsız.

Büyüklük eşiği: 6.478  |  test: 24 örnek, gerçek pozitif: 0, tahmin pozitif: 0
  Testte tek sınıf var; bu eşikte sınıflandırma metrikleri anlamsız.
```

**XgboostTests.ipynb**

```
threshold_report(test_frame_xgb, default_thresholds(train_xgb))
Büyüklük eşiği: 1.621  |  test: 9 örnek, gerçek pozitif: 8, tahmin pozitif: 1
  Accuracy=0.222 (çoğunluk sınıfı referansı: 0.889)
  Precision=1.000, Recall=0.125, F1=0.222 (hep 'büyük' de referansı F1: 0.941)
  Confusion matrix: TN=1, FP=0, FN=7, TP=1

Büyüklük eşiği: 1.661  |  test: 9 örnek, gerçek pozitif: 7, tahmin pozitif: 0
  Accuracy=0.222 (çoğunluk sınıfı referansı: 0.778)
  Precision=0.000, Recall=0.000, F1=0.000 (hep 'büyük' de referansı F1: 0.875)
  Confusion matrix: TN=2, FP=0, FN=7, TP=0

Büyüklük eşiği: 1.696  |  test: 9 örnek, gerçek pozitif: 6, tahmin pozitif: 0
  Accuracy=0.333 (çoğunluk sınıfı referansı: 0.667)
  Precision=0.000, Recall=0.000, F1=0.000 (hep 'büyük' de referansı F1: 0.800)
  Confusion matrix: TN=3, FP=0, FN=6, TP=0

threshold_report(test_frame_xgb_2, default_thresholds(train_xgb_2))
Büyüklük eşiği: 6.078  |  test: 24 örnek, gerçek pozitif: 0, tahmin pozitif: 0
  Testte tek sınıf var; bu eşikte sınıflandırma metrikleri anlamsız.

Büyüklük eşiği: 6.255  |  test: 24 örnek, gerçek pozitif: 0, tahmin pozitif: 0
  Testte tek sınıf var; bu eşikte sınıflandırma metrikleri anlamsız.

Büyüklük eşiği: 6.478  |  test: 24 örnek, gerçek pozitif: 0, tahmin pozitif: 0
  Testte tek sınıf var; bu eşikte sınıflandırma metrikleri anlamsız.
```

**gelismisdeneme.ipynb**

```
threshold_report(test_frame_mlp, default_thresholds(train_mlp))
Büyüklük eşiği: 1.618  |  test: 57 örnek, gerçek pozitif: 47, tahmin pozitif: 47
  Accuracy=0.754 (çoğunluk sınıfı referansı: 0.825)
  Precision=0.851, Recall=0.851, F1=0.851 (hep 'büyük' de referansı F1: 0.904)
  Confusion matrix: TN=3, FP=7, FN=7, TP=40

Büyüklük eşiği: 1.704  |  test: 57 örnek, gerçek pozitif: 31, tahmin pozitif: 4
  Accuracy=0.491 (çoğunluk sınıfı referansı: 0.544)
  Precision=0.750, Recall=0.097, F1=0.171 (hep 'büyük' de referansı F1: 0.705)
  Confusion matrix: TN=25, FP=1, FN=28, TP=3

Büyüklük eşiği: 1.79  |  test: 57 örnek, gerçek pozitif: 24, tahmin pozitif: 0
  Accuracy=0.579 (çoğunluk sınıfı referansı: 0.579)
  Precision=0.000, Recall=0.000, F1=0.000 (hep 'büyük' de referansı F1: 0.593)
  Confusion matrix: TN=33, FP=0, FN=24, TP=0
```

**denemeRNN.ipynb**

```
threshold_report(test_frame, default_thresholds(train_data))
Büyüklük eşiği: 1.624  |  test: 57 örnek, gerçek pozitif: 46, tahmin pozitif: 0
  Accuracy=0.193 (çoğunluk sınıfı referansı: 0.807)
  Precision=0.000, Recall=0.000, F1=0.000 (hep 'büyük' de referansı F1: 0.893)
  Confusion matrix: TN=11, FP=0, FN=46, TP=0

Büyüklük eşiği: 1.716  |  test: 57 örnek, gerçek pozitif: 31, tahmin pozitif: 0
  Accuracy=0.456 (çoğunluk sınıfı referansı: 0.544)
  Precision=0.000, Recall=0.000, F1=0.000 (hep 'büyük' de referansı F1: 0.705)
  Confusion matrix: TN=26, FP=0, FN=31, TP=0

Büyüklük eşiği: 1.827  |  test: 57 örnek, gerçek pozitif: 22, tahmin pozitif: 0
  Accuracy=0.614 (çoğunluk sınıfı referansı: 0.614)
  Precision=0.000, Recall=0.000, F1=0.000 (hep 'büyük' de referansı F1: 0.557)
  Confusion matrix: TN=35, FP=0, FN=22, TP=0

threshold_report(test_frame_2, default_thresholds(train_data))
Büyüklük eşiği: 1.624  |  test: 57 örnek, gerçek pozitif: 46, tahmin pozitif: 0
  Accuracy=0.193 (çoğunluk sınıfı referansı: 0.807)
  Precision=0.000, Recall=0.000, F1=0.000 (hep 'büyük' de referansı F1: 0.893)
  Confusion matrix: TN=11, FP=0, FN=46, TP=0

Büyüklük eşiği: 1.716  |  test: 57 örnek, gerçek pozitif: 31, tahmin pozitif: 0
  Accuracy=0.456 (çoğunluk sınıfı referansı: 0.544)
  Precision=0.000, Recall=0.000, F1=0.000 (hep 'büyük' de referansı F1: 0.705)
  Confusion matrix: TN=26, FP=0, FN=31, TP=0

Büyüklük eşiği: 1.827  |  test: 57 örnek, gerçek pozitif: 22, tahmin pozitif: 0
  Accuracy=0.614 (çoğunluk sınıfı referansı: 0.614)
  Precision=0.000, Recall=0.000, F1=0.000 (hep 'büyük' de referansı F1: 0.557)
  Confusion matrix: TN=35, FP=0, FN=22, TP=0

threshold_report(test_frame_3, default_thresholds(train_data))
Büyüklük eşiği: 6.078  |  test: 24 örnek, gerçek pozitif: 0, tahmin pozitif: 0
  Testte tek sınıf var; bu eşikte sınıflandırma metrikleri anlamsız.

Büyüklük eşiği: 6.255  |  test: 24 örnek, gerçek pozitif: 0, tahmin pozitif: 0
  Testte tek sınıf var; bu eşikte sınıflandırma metrikleri anlamsız.

Büyüklük eşiği: 6.478  |  test: 24 örnek, gerçek pozitif: 0, tahmin pozitif: 0
  Testte tek sınıf var; bu eşikte sınıflandırma metrikleri anlamsız.

threshold_report(test_frame_4, default_thresholds(train_data))
Büyüklük eşiği: 6.078  |  test: 24 örnek, gerçek pozitif: 0, tahmin pozitif: 0
  Testte tek sınıf var; bu eşikte sınıflandırma metrikleri anlamsız.

Büyüklük eşiği: 6.255  |  test: 24 örnek, gerçek pozitif: 0, tahmin pozitif: 0
  Testte tek sınıf var; bu eşikte sınıflandırma metrikleri anlamsız.

Büyüklük eşiği: 6.478  |  test: 24 örnek, gerçek pozitif: 0, tahmin pozitif: 0
  Testte tek sınıf var; bu eşikte sınıflandırma metrikleri anlamsız.
```

---

## 3. Problemin yeniden tanımlanması: eski versiyon ne tahmin ediyordu, 0.82 veren versiyon ne tahmin ediyor

1\. ve 2. bölümdeki düzeltmeler kodu doğru hale getirdi ama modeller yine bir şey öğrenemedi. Sebep kodda değil, tahmin edilmeye çalışılan şeydeydi. Bu bölüm, eski hedef ile `BolgeselRF.ipynb` içindeki yeni hedefin farkını anlatır.

### 3.1 Eski versiyon: "Gelecek hafta dünyadaki tüm depremlerin ortalaması ne olacak?"

Bir haftada dünyanın her yerinde kaydedilen ~1.800 depremin tamamı tek bir satıra indiriliyordu:

| Hafta | Ortalama büyüklük | Ortalama derinlik | Ortalama enlem | Ortalama boylam |
|---|---|---|---|---|
| 1 | 1.60 | 19.7 km | 37.4° | −114.9° |
| 2 | 1.68 | 22.7 km | 37.8° | −113.0° |
| 3 | ? | ? | ? | ? |

Model 1. ve 2. haftaya bakıp 3. haftanın bu dört ortalamasını tahmin ediyordu.

- **Girdi:** önceki haftaların dünya ortalamaları
- **Çıktı:** gelecek haftanın dünya ortalaması (4 sayı)
- **Veri:** 42 satır

Sorunu: bu sayılar bir olaya karşılık gelmiyor.

- **Ortalama büyüklük neredeyse sabit.** Kayıtların büyük çoğunluğu 1–2 büyüklüğünde küçük sarsıntılar. 1.799 küçük depremin ortalaması 1.65 ise, o hafta 7.0 büyüklüğünde bir deprem de olsa ortalama 1.653 olur. Yılın en önemli olayı ortalamayı binde 3 oynatır. Haftalık ortalamalar hep 1.6–1.7 bandında kalır; aradaki küçük oynama bir örüntüden değil, o hafta hangi bölgede kaç küçük deprem kaydedildiğinden gelir.
- **Ortalama konum gerçek bir yer değil.** USGS verisi küresel. Aynı hafta Kaliforniya'da (boylam −120) ve Japonya'da (boylam +140) deprem olursa ortalama boylam +10 çıkar; orada deprem olmamıştır.

Model en iyi ihtimalle "hep ortalamayı söyle" stratejisini öğrenebilir. 2. bölümdeki sonuçlar tam olarak bunu gösteriyor: train R² ≈ 0, test R² negatif.

### 3.2 Yeni versiyon: "Gelecek hafta bu bölgedeki en büyük deprem kaç olacak?"

Dünya 1°×1° karelere bölünüyor (yaklaşık 100 km × 100 km) ve her kare ayrı ayrı takip ediliyor. Aşağıdaki sayılar yapıyı göstermek için örnektir:

| Bölge | Hafta | Deprem sayısı | En büyük deprem | … | **Gelecek hafta en büyük** |
|---|---|---|---|---|---|
| Kaliforniya'da bir kare (36.5°, −117.5°) | 5 | 48 | 2.9 | … | **3.1** |
| Aynı kare | 6 | 52 | 3.1 | … | **2.7** |
| Japonya açıklarında bir kare (38.5°, 142.5°) | 5 | 2 | 4.8 | … | **5.1** |

- **Girdi:** o karenin son haftalardaki deprem sayısı, en büyük ve ortalama büyüklüğü, derinliği, geçmiş profili, komşu karelerin durumu, koordinatı
- **Çıktı:** o karede gelecek haftanın en büyük depreminin büyüklüğü (1 sayı)
- **Veri:** 8.044 satır (1.164 kare)

Hedef, bir sonraki haftada o karede en az bir deprem kaydı olan satırlar için tanımlıdır.

### 3.3 Yan yana

| | Eski | Yeni (bölgesel) |
|---|---|---|
| Soru | Dünya ortalaması ne olacak? | Bu bölgenin en büyük depremi kaç olacak? |
| Bakılan yer | Tüm dünya, tek satırda | Her bölge ayrı |
| Hedefin türü | Ortalama | En büyük değer |
| Hedef sayısı | 4 (büyüklük, derinlik, enlem, boylam) | 1 (büyüklük) |
| Hedefin aralığı | ~1.6 – 1.7 | ~0.5 – 7 |
| Satır sayısı (Dataset 1) | 42 | 8.044 |
| Random Forest test R² (Dataset 1) | −2.13 | 0.82 |

### 3.4 Bölgesel Random Forest sonuçları

Kronolojik bölme: ilk %80 dönem train, son %20 test. Referanslar: persistence (gelecek = şimdiki), hücre ortalaması (her karenin train'deki ortalaması), global ortalama.

**Dataset 1 (USGS 2022, 1°×1° kare × hafta): 1.164 kare, train 6.160, test 1.884 satır**

| Model | Veri | R² | MAE | MSE | RMSE |
|---|---|---|---|---|---|
| Random Forest | Train | 0.834 | 0.398 | 0.309 | 0.556 |
| Random Forest | Test | 0.817 | 0.427 | 0.355 | 0.596 |
| Persistence (referans) | Test | 0.694 | 0.563 | 0.594 | 0.771 |
| Hücre ortalaması (referans) | Test | 0.632 | 0.583 | 0.714 | 0.845 |
| Global ortalama (referans) | Test | −0.041 | 1.192 | 2.023 | 1.422 |

Test, eşik bazlı sınıflandırma (eşikler train hedef dağılımının %75 ve %90'lık dilimleri):

| Model | Eşik | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| Random Forest | ≥3.98 | 0.971 | 0.991 | 0.921 | 0.955 | 0.988 |
| Random Forest | ≥4.70 | 0.879 | 0.619 | 0.528 | 0.570 | 0.924 |
| Persistence | ≥3.98 | 0.956 | 0.932 | 0.934 | 0.933 | 0.976 |
| Persistence | ≥4.70 | 0.855 | 0.521 | 0.524 | 0.523 | 0.902 |

**Dataset 2 (M≥5.5, 10°×10° kare × yıl, 1973 sonrası): 161 kare, train 3.232, test 905 satır**

| Model | Veri | R² | MAE | MSE | RMSE |
|---|---|---|---|---|---|
| Random Forest | Train | 0.443 | 0.329 | 0.185 | 0.430 |
| Random Forest | Test | 0.252 | 0.381 | 0.249 | 0.499 |
| Hücre ortalaması (referans) | Test | 0.236 | 0.377 | 0.254 | 0.504 |
| Global ortalama (referans) | Test | −0.002 | 0.460 | 0.334 | 0.578 |
| Persistence (referans) | Test | −0.403 | 0.501 | 0.467 | 0.684 |

Dataset 2'de sonuç sınırlı: o katalog yalnızca M≥5.5 depremleri içerdiği için bir karede yılın en büyük depremi dar bir aralıkta oynuyor ve model hücre ortalaması referansını çok az geçiyor.

Tam tablolar (train tarafındaki sınıflandırma metrikleri dahil) `BolgeselRF.ipynb` çıktısında ve `results/bolgesel_rf_dataset1.json`, `results/bolgesel_rf_dataset2.json` dosyalarında.

### 3.5 0.82 neyi başarıyor, neyi başarmıyor

**Başardığı:** Bir bölge için "gelecek hafta burada en büyük deprem yaklaşık şu büyüklükte olur" tahminini ortalama 0.43 büyüklük birimi hatayla veriyor; train ve test skorları neredeyse aynı (ezber yok) ve üç referansı da geçiyor.

**Bunu nasıl yapıyor:** Büyük ölçüde her bölgenin kendi karakterini öğrenerek. Yoğun sismograf ağıyla izlenen bir karede küçük depremler de kaydediliyor ve haftanın en büyüğü genelde 2–3 oluyor. Okyanus ortasındaki bir karede sadece büyük depremler kaydedilebiliyor ve en büyük değer genelde 4–5. Model bu farkı ve bölgenin son haftalardaki hareketliliğini kullanıyor. En önemli özellikler de bunu gösteriyor: karenin geçmiş ortalama en büyük depremi, son 8 haftanın ortalaması ve komşu kareler.

**Başarmadığı:** "Şu tarihte şurada 7 büyüklüğünde deprem olacak" türünden bir öngörü. Bir bölgenin alışılmış seviyesinin çok üstündeki nadir büyük depremleri önceden yakalayamıyor (≥4.70 eşiğinde recall 0.53). Bu, deprem biliminin açık problemi.

**Özet:** Eski versiyon anlamı olmayan bir sayıyı tahmin etmeye çalışıyordu. Yeni versiyon gerçek bir soruyu cevaplıyor ve cevabı esas olarak "her bölgenin tipik deprem seviyesi" bilgisine dayanıyor.
