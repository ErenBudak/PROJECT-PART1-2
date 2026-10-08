# PROJECT-PART1-2 ve DEMO
Projede model kaydetme ve birkaç küçük noktada daha yapay zeka ve diğer kaynaklardan yardım aldım,yardım aldığım yerleri belirttim,NOT:Kayıtlı modeller pickle ve keras şeklinde kaydedildi,buradan load edilebilir.Feedback doğrultusunda konum ve eşiğe göre ölçüm eklendi.ÖNEMLİ NOT:Konum ve Eşik değerlendirmesi gelismisdeneme.ipynb de MLP ye eklendiği için MLPTESTS.ipynb ye eklemedim.
Projede yaptıklarım:
Demo ile biten kodlar:Feedback ve isteklere uygun şekilde demo yapıyor,tek verinin tahmini,teste erişim ve klasöroluşturup denem var.Direk klasörü github a  yükleyemediğim için klasörü oluşturup çalıştıracak şekilde bir kod yazdım,biraz da AI yardımıyla.
models klasörü:tüm modeller burada yer  alıyor(keras,pkl )
denemeRNN.ipynb:Bütün LSTM modelleri burada.LSTM ve LSTM+Attention testleri yer alıyor.
deepProject.ipynb:LSTM ve LSTM+Attention için gerekli classlar yer alıyor.
gelismisdeneme.ipynb:Günlük veriyle MLP + Outlier Detection (IsolationForest) denemesi.
MLProject.ipynb:Burada MLP,XGBoost ve Random Forest için gerekli Classlar yer alıyor.Temel preprocess tekniğimi bozmadım,sadece Xgboost ve MLP için gerekli yerlere eklemeler yaptım.
RandomForestTests.ipynb:RandomForest için dataset 1 ve 2 yi eğittim ve test ettim,MSE MAE VE R2 metrikleriyle sonuçlar elde ettim,ek bir model dahil.
XgBoostTests.ipynb:XgBoost için aynı şeyleri yaptım,çalışması çok uzun sürdüğünden ve benzer sorunlara benzer tepkiler vereceğinden ötürü ek versiyon koymadım.
MLPTests:En hızlı çalışan test,ek bir model da var.
usgs_main.csv:Dataset 1.Genel olarak haftalık timesteplerle kullanıldı
significant_earthquake_dataset_1900_2023.csv:Dataset 2,yıllık timesteplerle kullanıldı,ek versiyonlarda hafta olarak da denendi.
evaluation.py:Tüm modeller için ortak değerlendirme fonksiyonları (hizalı metrikler, train/test karşılaştırması, referans modeller, eşik ve konum analizi).
results klasörü:Her modelin train/test/referans metrikleri JSON olarak.
BolgeselRF.ipynb + regional.py:Bölge bazlı Random Forest. Dünya ızgara hücrelerine bölünür, her hücre için bir sonraki dönemin en büyük depremi tahmin edilir (veri azlığı ve ortalama alma sorunlarına karşı).
RAPOR.md:Yapılan düzeltmelerin sebepleriyle anlatımı ve yeniden eğitilen tüm modellerin train/test sonuçları.

## Düzeltmeler (fix/evaluation-and-tuning branch'i)

Kodu yeniden incelediğimde sonuçları geçersiz kılan birkaç hata buldum. Bu branch'te hepsi düzeltildi ve tüm modeller yeniden eğitildi.

1. **Hiperparametre araması hiç çalışmıyordu.** `GridSearchCV.fit(train)` çağrısında `y` verilmediği için `scoring='neg_mean_squared_error'` her denemede hata verip NaN dönüyordu (notebook çıktılarındaki `test scores are non-finite: [nan nan ...]` uyarısı). Bütün skorlar NaN olunca grid'deki ilk kombinasyon "en iyi" seçiliyordu. Artık hedefleri DataFrame'den okuyan özel bir scorer (`evaluation.ml_scorer`) ve `error_score='raise'` kullanılıyor.
2. **GridSearch'ün parametreleri modele ulaşmıyordu.** RF/XGBoost/MLP modelleri `__init__` içinde kuruluyordu; GridSearch parametreleri `set_params` ile sonradan verdiği için iç model hep varsayılan değerlerle eğitiliyordu (sadece `shiftnum` etkiliydi). Artık iç model `fit()` içinde kuruluyor.
3. **Tahminler yanlış satırlarla karşılaştırılıyordu.** Lag/pencere yüzünden model ilk `shiftnum` (veya `sequence_length`) satırı atıyor, değerlendirme kodu ise gerçek değerleri baştan alıyordu. Artık her model `predict_with_targets` ile (tahmin, gerçek, index) döndürüyor; test için train'in son satırları bağlam olarak ekleniyor, böylece testin tüm satırları tahmin ediliyor. LSTM'deki Optuna hedefi de aynı hatayı içeriyordu, o da düzeltildi.
4. **En güncel gözlem kullanılmıyordu.** Lag'ler `shift(1..k)` ile kurulup hedef `t+1` olduğu için `t` anı atlanıyor, model fiilen 2 adım ilerisini tahmin ediyordu. LSTM penceresinde de aynı boşluk vardı. Artık lag'ler `t, t-1, ...` ve pencere hedefi pencerenin son satırının bir sonraki dönemi.
5. **XGBoost ve RF hedef ölçeği.** XGBoost 4 hedef için tek bir başlangıç değeri (`base_score`, tüm hedeflerin ortalaması) kullanıyordu; magnitude bu değerden başlayıp gerçek seviyesine ulaşamıyordu. RF'nin çok çıktılı bölünme kriteri de büyük ölçekli hedeflere (depth, longitude) göre karar veriyordu. İkisinde de hedefler artık standartlaştırılıyor.
6. **Outlier detection sızıntısı.** gelismisdeneme'de IsolationForest test dahil tüm veriye uygulanıyor ve aykırı günler silinerek zaman çizelgesinde boşluk açılıyordu. Artık sadece train'e fit ediliyor ve aykırı değerler interpolasyonla dolduruluyor (LSTM+Attention'da da aynı). RobustScaler adımı kaldırıldı: ürettiği `*_robust` sütunları modele hiç girmiyordu.
7. **Overfit görünmüyordu.** Train skoru ve iki referans model (persistence: "gelecek dönem = bu dönem", train ortalaması) artık her modelde raporlanıyor.
8. **Eşik analizi anlamsızdı.** Sabit eşikler (ör. 2.0, 6.0) dönem ortalamalarının hiç ulaşmadığı değerlerdi; testte tek sınıf kalınca accuracy=1.000 çıkıyordu. Eşikler artık train'deki `futuremag` dağılımından (medyan, %75, %90) seçiliyor ve çoğunluk sınıfı referansıyla karşılaştırılıyor.

