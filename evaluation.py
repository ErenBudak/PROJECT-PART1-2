"""Tüm modeller (RF, XGBoost, MLP, LSTM) için ortak değerlendirme fonksiyonları.

Eski notebook'lardaki değerlendirme kodunda üç sorun vardı:
1. Model, lag/pencere yüzünden ilk `shiftnum` (veya `sequence_length`) satırı atıyor ama
   gerçek değerler baştan alınıyordu -> tahminler yanlış satırlarla karşılaştırılıyordu.
2. GridSearchCV'ye `y` verilmediği için `scoring='neg_mean_squared_error'` her denemede
   hata verip NaN dönüyordu -> grid'deki ilk kombinasyon "en iyi" seçiliyordu.
3. Train skoru ve basit referans (baseline) modeller raporlanmadığı için overfit görünmüyordu.

Buradaki fonksiyonlar her modelin `predict_with_targets(df)` metodunu kullanır. Bu metod
(tahmin, gerçek hedef, satır index'i) üçlüsünü döndürür, yani hizalama modelin kendisinden gelir.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

FEATURE_COLS = ['mag', 'depth', 'latitude', 'longitude']
TARGET_COLS = ['futuremag', 'futuredepth', 'futurelat', 'futurelon']
TARGET_LABELS = ['Magnitude', 'Depth', 'Latitude', 'Longitude']
PRED_COLS = [f'pred_{c}' for c in TARGET_COLS]


def context_length(model):
    """Modelin bir tahmin için kaç geçmiş satıra ihtiyaç duyduğu."""
    if hasattr(model, 'shiftnum'):
        return model.shiftnum
    return model.sequence_length


def normalized_mse(y_true, y_pred, scale):
    """Her hedefin MSE'sini train'deki varyansına bölüp ortalar.

    Hedeflerin birimleri çok farklı (magnitude ~1, depth ~yüzlerce km). Ham MSE'de depth ve
    longitude baskın çıkar; bu ölçüt dört hedefe eşit ağırlık verir.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(((y_true - y_pred) / scale) ** 2))


def target_scale(Y):
    """normalized_mse için hedef başına ölçek (std). Sabit hedefte 0'a bölmeyi engeller."""
    std = np.asarray(Y, dtype=float).std(axis=0)
    return np.where(std > 1e-12, std, 1.0)


def ml_scorer(estimator, X, y=None):
    """GridSearchCV için scorer. Hedefler DataFrame'in içinde olduğu için `y` kullanılmaz.

    Büyük skor = iyi olduğu için negatif normalize MSE döner.
    """
    y_pred, y_true, _ = estimator.predict_with_targets(X)
    return -normalized_mse(y_true, y_pred, estimator.target_scale_)


def predict_aligned(model, df, context_df=None):
    """Tahminleri doğru satırlarla eşleşmiş bir DataFrame olarak döndürür.

    `context_df` verilirse (ör. test için train), onun son satırları geçmiş olarak eklenir.
    Böylece testin ilk satırları da tahmin edilebilir; sonuçta sadece `df` satırları kalır.
    """
    full = df
    if context_df is not None:
        full = pd.concat([context_df.tail(context_length(model)), df])
    y_pred, y_true, index = model.predict_with_targets(full)
    frame = pd.DataFrame(np.asarray(y_true, dtype=float), index=index, columns=TARGET_COLS)
    frame[PRED_COLS] = np.asarray(y_pred, dtype=float)
    frame[FEATURE_COLS] = full.loc[index, FEATURE_COLS].values
    return frame[frame.index.isin(df.index)]


def _metrics(y_true, y_pred):
    return {
        'mse': float(mean_squared_error(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
    }


def evaluate_model(model, train, test, name, save_dir='results'):
    """Train/test metriklerini ve iki referans modeli yan yana raporlar.

    Referanslar:
      - Persistence: "gelecek dönem = bu dönem" (futuremag = mag)
      - Train ortalaması: her zaman train'deki hedef ortalamasını tahmin et
    Bir model bunları geçemiyorsa geçmişten anlamlı bir şey öğrenmemiş demektir.
    """
    train_frame = predict_aligned(model, train)
    test_frame = predict_aligned(model, test, context_df=train)
    train_mean = train[TARGET_COLS].mean().values

    results = {'name': name, 'n_train': len(train_frame), 'n_test': len(test_frame), 'targets': {}}
    rows = []
    for target, pred, feature, label, mean in zip(TARGET_COLS, PRED_COLS, FEATURE_COLS, TARGET_LABELS, train_mean):
        r = {
            'train': _metrics(train_frame[target], train_frame[pred]),
            'test': _metrics(test_frame[target], test_frame[pred]),
            'persistence_test': _metrics(test_frame[target], test_frame[feature]),
            'train_mean_test': _metrics(test_frame[target], np.full(len(test_frame), mean)),
        }
        results['targets'][target] = r
        rows.append({
            'Hedef': label,
            'Train R²': r['train']['r2'],
            'Test R²': r['test']['r2'],
            'Persistence R²': r['persistence_test']['r2'],
            'Train-ort. R²': r['train_mean_test']['r2'],
            'Test MAE': r['test']['mae'],
            'Persistence MAE': r['persistence_test']['mae'],
            'Train-ort. MAE': r['train_mean_test']['mae'],
        })

    scale = target_scale(train_frame[TARGET_COLS].values)
    results['test_normalized_mse'] = normalized_mse(test_frame[TARGET_COLS], test_frame[PRED_COLS], scale)
    results['persistence_normalized_mse'] = normalized_mse(test_frame[TARGET_COLS], test_frame[FEATURE_COLS], scale)
    results['train_mean_normalized_mse'] = normalized_mse(
        test_frame[TARGET_COLS], np.tile(train_mean, (len(test_frame), 1)), scale)

    print(f"\n=== {name} | train: {results['n_train']} satır, test: {results['n_test']} satır ===")
    print(pd.DataFrame(rows).set_index('Hedef').round(3).to_string())
    print(f"Normalize test MSE -> model: {results['test_normalized_mse']:.3f}, "
          f"persistence: {results['persistence_normalized_mse']:.3f}, "
          f"train ortalaması: {results['train_mean_normalized_mse']:.3f}  (düşük = iyi)")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{name.replace(' ', '_').lower()}.json"), 'w') as f:
            json.dump(results, f, indent=2)
    return test_frame, results


def default_thresholds(train, quantiles=(0.5, 0.75, 0.9)):
    """Eşikleri train'deki futuremag dağılımından seçer.

    Sabit eşikler (ör. 2.0 veya 6.0) dönem ortalamalarının hiç ulaşmadığı değerlerdi;
    testte tek sınıf kalınca accuracy=1.000 çıkıyordu ama hiçbir şey ölçmüyordu.
    """
    return [round(float(train['futuremag'].quantile(q)), 3) for q in quantiles]


def threshold_report(test_frame, thresholds):
    """Tahmin edilen magnitude'u eşiğe göre "büyük / değil" sınıflandırmasına çevirir."""
    summary = {}
    for threshold in thresholds:
        actual = (test_frame['futuremag'] >= threshold).astype(int).values
        predicted = (test_frame['pred_futuremag'] >= threshold).astype(int).values
        n, n_pos = len(actual), int(actual.sum())
        print(f"\nBüyüklük eşiği: {threshold}  |  test: {n} örnek, gerçek pozitif: {n_pos}, "
              f"tahmin pozitif: {int(predicted.sum())}")
        if n_pos == 0 or n_pos == n:
            print("  Testte tek sınıf var; bu eşikte sınıflandırma metrikleri anlamsız.")
            summary[threshold] = {'single_class': True, 'n': n, 'n_pos': n_pos}
            continue
        tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / n
        majority_acc = max(n_pos, n - n_pos) / n
        always_pos_f1 = 2 * n_pos / (n + n_pos)
        print(f"  Accuracy={accuracy:.3f} (çoğunluk sınıfı referansı: {majority_acc:.3f})")
        print(f"  Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f} "
              f"(hep 'büyük' de referansı F1: {always_pos_f1:.3f})")
        print(f"  Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        summary[threshold] = {'accuracy': accuracy, 'majority_accuracy': majority_acc, 'precision': precision,
                              'recall': recall, 'f1': f1, 'always_positive_f1': always_pos_f1,
                              'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    return summary


def location_report(test_frame, threshold, grid_deg=1.0, min_count=2, max_locations=10):
    """Bölge bazlı kontrol: her hücrede eşik üstü dönem var mı, model bunu bildi mi?"""
    frame = test_frame.copy()
    frame['lat_group'] = np.round(frame['latitude'] / grid_deg) * grid_deg
    frame['lon_group'] = np.round(frame['longitude'] / grid_deg) * grid_deg
    groups = frame.groupby(['lat_group', 'lon_group'])
    sizes = groups.size()
    valid = sizes[sizes >= min_count].index
    print(f"\nEşik {threshold}, {grid_deg}° hücreler: {len(sizes)} bölge, "
          f"en az {min_count} veri olan: {len(valid)}")
    if len(valid) == 0:
        print("  Yeterli veri olan bölge yok; bölge bazlı değerlendirme yapılamıyor.")
        return {'n_regions': 0}
    correct = 0
    for lat, lon in valid[:max_locations]:
        data = groups.get_group((lat, lon))
        has_actual = bool((data['futuremag'] >= threshold).any())
        has_pred = bool((data['pred_futuremag'] >= threshold).any())
        correct += has_actual == has_pred
        print(f"  ({lat:.0f}°, {lon:.0f}°) veri={len(data)} gerçek={'var' if has_actual else 'yok'} "
              f"tahmin={'var' if has_pred else 'yok'} {'✓' if has_actual == has_pred else '✗'}")
    n = min(len(valid), max_locations)
    print(f"  Doğru: {correct}/{n}")
    return {'n_regions': n, 'correct': correct}
