# PROJECT-PART1
Projede model kaydetme ve birkaç küçük noktada daha yapay zeka ve diğer kaynaklardan yardım aldım,yardım aldığım yerleri belirttim,NOT:Kayıtlı modeller pickle şeklinde kaydedildi,buradan load edilebilir.
Projede yaptıklarım:
models klasörü:9 model de pkl dosyası şeklinde kaydedildi.
MLPProvider.ipynb:En iyi model,RobustScaler+Outlier Detection ile iyileşti
MLProject.ipynb:Burada MLP,XGBoost ve Random Forest için gerekli Classlar yer alıyor.Temel preprocess tekniğimi bozmadım,sadece Xgboost ve MLP için gerekli yerlere eklemeler yaptım.
RandomForestTests.ipynb:RandomForest için dataset 1 ve 2 yi eğittim ve test ettim,MSE MAE VE R2 metrikleriyle sonuçlar elde ettim,ek bir model dahil.
XgBoostTests.ipynb:XgBoost için aynı şeyleri yaptım,çalışması çok uzun sürdüğünden ve benzer sorunlara benzer tepkiler vereceğinden ötürü ek versiyon koymadım.
MLPTests:En hızlı çalışan test,ek bir model da var.
usgs_main.csv:Dataset 1.Genel olarak haftalık timesteplerle kullanıldı
significant_earthquake_dataset_1900_2023.csv:Dataset 2,yıllık timesteplerle kullanıldı,ek versiyonlarda hafta olarak da denendi.

