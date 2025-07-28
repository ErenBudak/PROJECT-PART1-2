# PROJECT-PART1-2 ve DEMO
Projede model kaydetme ve birkaç küçük noktada daha yapay zeka ve diğer kaynaklardan yardım aldım,yardım aldığım yerleri belirttim,NOT:Kayıtlı modeller pickle ve keras şeklinde kaydedildi,buradan load edilebilir.Feedback doğrultusunda konum ve eşiğe göre ölçüm eklendi.ÖNEMLİ NOT:Konum ve Eşik değerlendirmesi gelismisdeneme.ipynb de MLP ye eklendiği için MLPTESTS.ipynb ye eklemedim.
Projede yaptıklarım:
Demo ile biten kodlar:Feedback ve isteklere uygun şekilde demo yapıyor,tek verinin tahmini,teste erişim ve klasöroluşturup denem var.Direk klasörü github a  yükleyemediğim için klasörü oluşturup çalıştıracak şekilde bir kod yazdım,biraz da AI yardımıyla.
models klasörü:tüm modeller burada yer  alıyor(keras,pkl )
denemeRNN.ipynb:Bütün LSTM modelleri burada.LSTM ve LSTM+Attention testleri yer alıyor.
deepProject.ipynb:LSTM ve LSTM+Attention için gerekli classlar yer alıyor.
gelismisdeneme.ipynb:En iyi model,RobustScaler+Outlier Detection ile iyileşti.
MLProject.ipynb:Burada MLP,XGBoost ve Random Forest için gerekli Classlar yer alıyor.Temel preprocess tekniğimi bozmadım,sadece Xgboost ve MLP için gerekli yerlere eklemeler yaptım.
RandomForestTests.ipynb:RandomForest için dataset 1 ve 2 yi eğittim ve test ettim,MSE MAE VE R2 metrikleriyle sonuçlar elde ettim,ek bir model dahil.
XgBoostTests.ipynb:XgBoost için aynı şeyleri yaptım,çalışması çok uzun sürdüğünden ve benzer sorunlara benzer tepkiler vereceğinden ötürü ek versiyon koymadım.
MLPTests:En hızlı çalışan test,ek bir model da var.
usgs_main.csv:Dataset 1.Genel olarak haftalık timesteplerle kullanıldı
significant_earthquake_dataset_1900_2023.csv:Dataset 2,yıllık timesteplerle kullanıldı,ek versiyonlarda hafta olarak da denendi.

