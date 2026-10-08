"""Bölge bazlı (hücre x dönem) deprem veri seti ve değerlendirme fonksiyonları.

Eski kurulumda tüm dünyadaki depremlerin haftalık/yıllık ORTALAMASI alınıyordu:
  - 75 bin kayıt 42 satıra iniyordu (makine öğrenmesi için çok az),
  - enlem/boylam ortalaması gerçek bir konuma karşılık gelmiyordu,
  - ortalama büyüklük büyük depremleri bastırıyordu.

Burada dünya ızgara hücrelerine bölünür ve her hücre için ayrı bir zaman serisi kurulur.
Her satır = (hücre, dönem). Hedef: aynı hücrede BİR SONRAKİ dönemin en büyük depremi.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)

CELL_COLS = ['cell_lat', 'cell_lon']
# Her dönem için hücre bazında hesaplanan ve lag'leri alınan sütunlar
DYNAMIC_COLS = ['log_count', 'mean_mag', 'max_mag', 'std_mag', 'mean_depth', 'n_big', 'log_energy']
TARGET_COL = 'future_max_mag'


def build_panel(df, cell_deg, freq, min_active, big_mag):
    """Deprem kayıtlarını (hücre, dönem) tablosuna çevirir.

    `min_active`: dönemlerin en az bu oranında deprem kaydı olan hücreler tutulur.
    `big_mag`: "büyük deprem" sayısı özelliği için eşik.
    İlk ve son dönem yarım olduğu için atılır. Deprem olmayan dönemlerde sayı 0 olur,
    büyüklük/derinlik istatistikleri hücrenin son bilinen değeriyle doldurulur (ileriye doğru;
    gelecekten bilgi sızmaz).
    """
    df = df.dropna(subset=['latitude', 'longitude', 'depth', 'mag', 'time']).copy()
    df['cell_lat'] = np.floor(df['latitude'] / cell_deg) * cell_deg + cell_deg / 2
    df['cell_lon'] = np.floor(df['longitude'] / cell_deg) * cell_deg + cell_deg / 2
    time = pd.to_datetime(df['time'])
    if time.dt.tz is not None:
        time = time.dt.tz_localize(None)
    df['period'] = time.dt.to_period(freq)

    periods = pd.period_range(df['period'].min(), df['period'].max(), freq=freq)[1:-1]
    df = df[df['period'].isin(periods)]
    df['big'] = (df['mag'] >= big_mag).astype(int)
    # Sismik enerji büyüklükle üstel artar (log10 E ~ 1.5 M)
    df['energy'] = 10 ** (1.5 * df['mag'])

    agg = df.groupby(CELL_COLS + ['period']).agg(
        count=('mag', 'size'), mean_mag=('mag', 'mean'), max_mag=('mag', 'max'), std_mag=('mag', 'std'),
        mean_depth=('depth', 'mean'), n_big=('big', 'sum'), energy=('energy', 'sum'))

    activity = agg.groupby(level=CELL_COLS).size() / len(periods)
    cells = activity[activity >= min_active].index
    full_index = pd.MultiIndex.from_tuples(
        [(lat, lon, p) for lat, lon in cells for p in periods], names=CELL_COLS + ['period'])
    panel = agg.reindex(full_index).reset_index()

    for col in ['count', 'n_big', 'energy', 'std_mag']:
        panel[col] = panel[col].fillna(0)
    by_cell = panel.groupby(CELL_COLS, sort=False)
    # Hedef doldurulmamış değerden hesaplanır: sonraki dönemde deprem yoksa hedef tanımsızdır (satır atılır)
    panel[TARGET_COL] = by_cell['max_mag'].shift(-1)
    stat_cols = ['mean_mag', 'max_mag', 'mean_depth']
    panel[stat_cols] = by_cell[stat_cols].ffill()
    panel['log_count'] = np.log1p(panel['count'])
    panel['log_energy'] = np.log10(panel['energy'] + 1)
    panel['period_idx'] = panel['period'].map({p: i for i, p in enumerate(periods)})
    return panel


def _neighbor_mean(panel, cols):
    """Her (hücre, dönem) için 8 komşu hücrenin aynı dönemdeki ortalaması (tutulan hücreler arasından)."""
    step = np.min(np.diff(np.sort(panel['cell_lat'].unique())))
    lookup = panel.set_index(CELL_COLS + ['period_idx'])[cols]
    total = np.zeros((len(panel), len(cols)))
    n = np.zeros((len(panel), len(cols)))
    for dlat in (-step, 0, step):
        for dlon in (-step, 0, step):
            if dlat == 0 and dlon == 0:
                continue
            keys = pd.MultiIndex.from_arrays([panel['cell_lat'] + dlat, panel['cell_lon'] + dlon, panel['period_idx']])
            values = lookup.reindex(keys).values
            found = ~np.isnan(values)
            total += np.where(found, values, 0)
            n += found
    return np.where(n > 0, total / np.maximum(n, 1), np.nan), n[:, 0]


def make_features(panel, max_lags):
    """Özellik tablosunu üretir. Bütün özellikler sadece şimdiki ve geçmiş dönemleri kullanır.

    - Lag'ler: her dinamik sütun için lag0 (şimdiki dönem) ... lag{max_lags-1}, hücre içinde hesaplanır.
    - Hücrenin geçmiş profili (`hist_*`): o ana kadarki ortalama/en büyük değerler ("bu bölge genelde nasıl").
    - Kayan pencereler (`roll*`): son 4 ve 8 dönemin özeti.
    - Komşular (`nb_*`): çevredeki 8 hücrenin şimdiki aktivitesi.
    """
    data = panel[CELL_COLS + ['period_idx', TARGET_COL]].copy()
    by_cell = panel.groupby(CELL_COLS, sort=False)
    for lag in range(max_lags):
        for col in DYNAMIC_COLS:
            data[f'{col}_lag{lag}'] = by_cell[col].shift(lag)

    def per_cell(series):
        return series.reset_index(level=CELL_COLS, drop=True)

    for col in ['log_count', 'max_mag', 'mean_mag', 'mean_depth']:
        data[f'hist_mean_{col}'] = per_cell(by_cell[col].expanding().mean())
    data['hist_max_max_mag'] = by_cell['max_mag'].cummax()
    data['hist_std_max_mag'] = per_cell(by_cell['max_mag'].expanding().std()).fillna(0)
    for window in (4, 8):
        if window <= max_lags:
            data[f'roll{window}_mean_max_mag'] = per_cell(by_cell['max_mag'].rolling(window, min_periods=1).mean())
            data[f'roll{window}_max_max_mag'] = per_cell(by_cell['max_mag'].rolling(window, min_periods=1).max())
            data[f'roll{window}_mean_log_count'] = per_cell(by_cell['log_count'].rolling(window, min_periods=1).mean())

    neighbors, n_neighbors = _neighbor_mean(panel, ['log_count', 'max_mag'])
    data['nb_log_count'] = np.nan_to_num(neighbors[:, 0], nan=0.0)
    # Komşusu olmayan hücrede komşu büyüklüğü yerine hücrenin kendi değeri kullanılır
    data['nb_max_mag'] = pd.Series(neighbors[:, 1], index=data.index).fillna(data['max_mag_lag0'])
    data['nb_count'] = n_neighbors
    return data.dropna().reset_index(drop=True)


def feature_columns(data):
    return [c for c in data.columns if c not in ('period_idx', TARGET_COL)]


def split_by_period(data, train_fraction=0.8):
    """Kronolojik bölme: ilk dönemler train, son dönemler test (tüm hücreler için aynı tarih)."""
    periods = np.sort(data['period_idx'].unique())
    cut = periods[int(train_fraction * len(periods))]
    return data[data['period_idx'] < cut], data[data['period_idx'] >= cut]


class PeriodSplit:
    """Panel veri için TimeSeriesSplit: validation her zaman train'den sonraki dönemlerden oluşur.

    Sıradan TimeSeriesSplit satır sırasına göre böler; satırlar hücreye göre sıralı olduğu için
    geleceği train'e, geçmişi validation'a koyardı. Burada bölme dönem numarasına (groups) göre yapılır.
    """

    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        groups = np.asarray(groups)
        periods = np.sort(np.unique(groups))
        fold = len(periods) // (self.n_splits + 1)
        for i in range(1, self.n_splits + 1):
            train_periods = periods[:i * fold]
            val_periods = periods[i * fold:(i + 1) * fold] if i < self.n_splits else periods[i * fold:]
            yield np.where(np.isin(groups, train_periods))[0], np.where(np.isin(groups, val_periods))[0]


class LagSelector(BaseEstimator, TransformerMixin):
    """Özelliklerden sadece ilk `n_lags` lag'i seçer; kaç dönem geriye bakılacağı GridSearch ile aranabilsin diye."""

    def __init__(self, n_lags=2):
        self.n_lags = n_lags

    def fit(self, X, y=None):
        self.columns_ = [c for c in X.columns if '_lag' not in c or int(c.rsplit('_lag', 1)[1]) < self.n_lags]
        return self

    def transform(self, X):
        return X[self.columns_]


def baseline_predictions(train, data):
    """Modelin geçmesi gereken basit referanslar (hepsi sadece train bilgisinden kurulur)."""
    cell_mean = train.groupby(CELL_COLS)[TARGET_COL].mean()
    global_mean = train[TARGET_COL].mean()
    keys = pd.MultiIndex.from_frame(data[CELL_COLS])
    return {
        # gelecek dönemin en büyüğü = bu dönemin en büyüğü
        'Persistence': data['max_mag_lag0'].values,
        # her zaman train'deki genel ortalama
        'Global ortalama': np.full(len(data), global_mean),
        # her hücre için o hücrenin train'deki ortalaması: "hangi bölge daha aktif" bilgisini içerir,
        # zaman bilgisini içermez. Train'de hiç görülmemiş hücrede genel ortalama kullanılır.
        'Hücre ortalaması': cell_mean.reindex(keys).fillna(global_mean).values,
    }


def regression_metrics(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    return {'R²': float(r2_score(y_true, y_pred)), 'MAE': float(mean_absolute_error(y_true, y_pred)),
            'MSE': mse, 'RMSE': float(np.sqrt(mse))}


def classification_metrics(y_true, y_pred, threshold):
    """Regresyon tahmini eşiğe göre "eşik üstü deprem olacak mı" sınıflandırmasına çevrilir.

    AUC'de tahmin edilen büyüklük skor olarak kullanılır (0.5 = rastgele, 1.0 = kusursuz sıralama).
    """
    actual = (np.asarray(y_true) >= threshold).astype(int)
    predicted = (np.asarray(y_pred) >= threshold).astype(int)
    both_classes = 0 < actual.sum() < len(actual)
    return {
        'Pozitif oranı': float(actual.mean()),
        'Accuracy': float(accuracy_score(actual, predicted)),
        'Precision': float(precision_score(actual, predicted, zero_division=0)),
        'Recall': float(recall_score(actual, predicted, zero_division=0)),
        'F1': float(f1_score(actual, predicted, zero_division=0)),
        'AUC': float(roc_auc_score(actual, y_pred)) if both_classes else float('nan'),
    }


def evaluate_regional(model_name, pred_train, pred_test, train, test, name, save_dir='results'):
    """Modelin train ve test metriklerini, referans modellerle birlikte tablo olarak verir."""
    y_train, y_test = train[TARGET_COL].values, test[TARGET_COL].values
    thresholds = {'%75': float(np.quantile(y_train, 0.75)), '%90': float(np.quantile(y_train, 0.90))}
    predictions = {(model_name, 'Train'): (y_train, pred_train), (model_name, 'Test'): (y_test, pred_test)}
    for split, data, y in (('Train', train, y_train), ('Test', test, y_test)):
        for baseline, pred in baseline_predictions(train, data).items():
            predictions[(baseline + ' (referans)', split)] = (y, pred)

    regression, classification, results = [], [], {}
    for (model, split), (y, pred) in predictions.items():
        reg = regression_metrics(y, pred)
        regression.append({'Model': model, 'Veri': split, **reg})
        results.setdefault(model, {})[split.lower()] = {'regression': reg, 'classification': {}}
        for label, threshold in thresholds.items():
            cls = classification_metrics(y, pred, threshold)
            classification.append({'Model': model, 'Veri': split, 'Eşik': f"≥{threshold:.2f} ({label})", **cls})
            results[model][split.lower()]['classification'][label] = cls

    regression = pd.DataFrame(regression).set_index(['Model', 'Veri']).round(3)
    classification = pd.DataFrame(classification).set_index(['Model', 'Veri', 'Eşik']).round(3)
    n_cells = int(pd.concat([train, test]).groupby(CELL_COLS).ngroups)
    print(f"\n=== {name} | {n_cells} hücre, train: {len(train)} satır, test: {len(test)} satır ===")
    print("\nRegresyon metrikleri (hedef: bir sonraki dönemin en büyük depremi)")
    print(regression.to_string())
    print("\nSınıflandırma metrikleri (eşikler: train hedef dağılımının %75 ve %90'lık dilimleri)")
    print(classification.to_string())
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{name.replace(' ', '_').lower()}.json"), 'w') as f:
            json.dump({'name': name, 'n_cells': n_cells, 'n_train': len(train), 'n_test': len(test),
                       'thresholds': thresholds, 'models': results}, f, indent=2)
    return regression, classification
