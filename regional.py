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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

DYNAMIC_COLS = ['count', 'mean_mag', 'max_mag', 'mean_depth']
STATIC_COLS = ['cell_lat', 'cell_lon']
TARGET_COL = 'future_max_mag'


def build_panel(df, cell_deg, freq, min_active=0.8):
    """Deprem kayıtlarını (hücre, dönem) tablosuna çevirir.

    Sadece dönemlerin en az `min_active` oranında deprem kaydı olan hücreler tutulur; neredeyse hiç
    deprem olmayan hücrelerde tahmin edilecek bir şey yok. İlk ve son dönem yarım olduğu için atılır.
    Deprem olmayan dönemlerde count=0 olur, büyüklük/derinlik istatistikleri hücrenin son bilinen
    değeriyle doldurulur (ileriye doğru; gelecekten bilgi sızmaz).
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

    agg = df.groupby(STATIC_COLS + ['period']).agg(
        count=('mag', 'size'), mean_mag=('mag', 'mean'), max_mag=('mag', 'max'), mean_depth=('depth', 'mean'))

    activity = agg.groupby(level=STATIC_COLS).size() / len(periods)
    cells = activity[activity >= min_active].index
    full_index = pd.MultiIndex.from_tuples(
        [(lat, lon, p) for lat, lon in cells for p in periods], names=STATIC_COLS + ['period'])
    panel = agg.reindex(full_index).reset_index()

    panel['count'] = panel['count'].fillna(0)
    by_cell = panel.groupby(STATIC_COLS, sort=False)
    # Hedef doldurulmamış değerden hesaplanır: sonraki dönemde deprem yoksa hedef tanımsızdır (satır atılır)
    panel[TARGET_COL] = by_cell['max_mag'].shift(-1)
    stat_cols = ['mean_mag', 'max_mag', 'mean_depth']
    panel[stat_cols] = by_cell[stat_cols].ffill()
    panel['period_idx'] = panel['period'].map({p: i for i, p in enumerate(periods)})
    return panel


def make_features(panel, max_lags):
    """Her dinamik sütun için lag0 (şimdiki dönem) ... lag{max_lags-1} sütunlarını üretir.

    Lag'ler hücre içinde hesaplanır; bir hücrenin geçmişi başka hücreyle karışmaz.
    """
    data = panel[STATIC_COLS + ['period_idx', TARGET_COL]].copy()
    by_cell = panel.groupby(STATIC_COLS, sort=False)
    for lag in range(max_lags):
        for col in DYNAMIC_COLS:
            data[f'{col}_lag{lag}'] = by_cell[col].shift(lag)
    # Hücrenin o ana kadarki geçmiş ortalamaları (sadece geçmiş ve şimdiki dönemler; gelecekten bilgi yok).
    # "Bu hücre genelde ne kadar aktif" bilgisini modele doğrudan verir.
    for col in ['count', 'max_mag']:
        data[f'hist_{col}'] = by_cell[col].expanding().mean().reset_index(level=STATIC_COLS, drop=True)
    return data.dropna().reset_index(drop=True)


def feature_columns(data):
    return STATIC_COLS + [c for c in data.columns if '_lag' in c or c.startswith('hist_')]


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


def baseline_predictions(train, test):
    """Modelin geçmesi gereken basit referanslar."""
    cell_mean = train.groupby(STATIC_COLS)[TARGET_COL].mean()
    global_mean = train[TARGET_COL].mean()
    keys = pd.MultiIndex.from_frame(test[STATIC_COLS])
    return {
        # gelecek dönemin en büyüğü = bu dönemin en büyüğü
        'Persistence': test['max_mag_lag0'].values,
        # her zaman train'deki genel ortalama
        'Global ortalama': np.full(len(test), global_mean),
        # her hücre için o hücrenin train'deki ortalaması: "hangi bölge daha aktif" bilgisini içerir,
        # zaman bilgisini içermez. Model bunu geçiyorsa zamandan da bir şey öğrenmiş demektir.
        'Hücre ortalaması': cell_mean.reindex(keys).fillna(global_mean).values,
    }


def _scores(y_true, y_pred, thresholds):
    scores = {
        'r2': float(r2_score(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    for name, threshold in thresholds.items():
        positive = (y_true >= threshold).astype(int)
        # AUC: tahmin edilen büyüklük, "eşik üstü deprem olacak" için skor olarak kullanılıyor.
        # 0.5 = rastgele, 1.0 = kusursuz. Sabit tahmin (global ortalama) her zaman 0.5 verir.
        scores[f'auc_{name}'] = float(roc_auc_score(positive, y_pred)) if 0 < positive.sum() < len(positive) else float('nan')
    return scores


def evaluate_regional(predictions, train, test, name, save_dir='results'):
    """Modelleri ve referansları aynı tabloda karşılaştırır.

    `predictions`: {model adı: (train tahmini, test tahmini)}
    """
    y_train, y_test = train[TARGET_COL].values, test[TARGET_COL].values
    thresholds = {'q75': float(np.quantile(y_train, 0.75)), 'q90': float(np.quantile(y_train, 0.90))}
    results = {'name': name, 'n_train': len(train), 'n_test': len(test),
               'n_cells': int(train.groupby(STATIC_COLS).ngroups), 'thresholds': thresholds, 'models': {}}
    rows = []
    for model, pred in baseline_predictions(train, test).items():
        s = _scores(y_test, pred, thresholds)
        results['models'][model] = {'test': s}
        rows.append({'Model': model, 'Train R²': np.nan, 'Test R²': s['r2'], 'Test MAE': s['mae'],
                     'Test RMSE': s['rmse'], 'AUC (≥%75)': s['auc_q75'], 'AUC (≥%90)': s['auc_q90']})
    for model, (pred_train, pred_test) in predictions.items():
        s = _scores(y_test, pred_test, thresholds)
        train_r2 = float(r2_score(y_train, pred_train))
        results['models'][model] = {'train_r2': train_r2, 'test': s}
        rows.append({'Model': model, 'Train R²': train_r2, 'Test R²': s['r2'], 'Test MAE': s['mae'],
                     'Test RMSE': s['rmse'], 'AUC (≥%75)': s['auc_q75'], 'AUC (≥%90)': s['auc_q90']})

    table = pd.DataFrame(rows).set_index('Model').round(3)
    print(f"\n=== {name} | {results['n_cells']} hücre, train: {len(train)} satır, test: {len(test)} satır ===")
    print(f"Eşikler (train hedef dağılımı): %75 = {thresholds['q75']:.2f}, %90 = {thresholds['q90']:.2f}")
    print(table.to_string())
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{name.replace(' ', '_').lower()}.json"), 'w') as f:
            json.dump(results, f, indent=2)
    return table, results
