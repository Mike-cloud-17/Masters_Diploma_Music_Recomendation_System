#!/usr/bin/env python3
# ml_classic_suite.py
# Запуск: python ml_classic_suite.py

import os, gc, time, datetime, warnings, joblib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model   import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import (RandomForestClassifier, ExtraTreesClassifier,
                                    AdaBoostClassifier, GradientBoostingClassifier,
                                    HistGradientBoostingClassifier)
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.svm            import LinearSVC, SVC
from sklearn.naive_bayes    import GaussianNB, MultinomialNB

# XGBoost
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────
def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")

def mem(df):
    mb = df.memory_usage(deep=True).sum()/1024/1024
    return f"{mb:,.1f} MB"

def get_metrics(name, y_true, y_score):
    y_pred = (y_score > .5).astype(int)
    return {
        'Model': name,
        'AUC': roc_auc_score(y_true, y_score),
        'ACC': accuracy_score(y_true, y_pred),
        'F1':  f1_score(y_true, y_pred),
        'Prec': precision_score(y_true, y_pred),
        'Rec':  recall_score(y_true, y_pred)
    }

# ────────────────────────────────────────────────────────────────────────
# Paths & settings
# ────────────────────────────────────────────────────────────────────────
folder     = 'training'            # или 'validation'
INPUT_DIR  = f'./input/{folder}'
MODEL_DIR  = './models'
os.makedirs(MODEL_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────
# 1) Feature importance: drop low-impact cols
# ────────────────────────────────────────────────────────────────────────
drop_cols = (pd.read_csv('./lgb_feature_importance.csv')
               .query('importance<85')['name']
               .tolist())
log(f"Feature-importance: {len(drop_cols)} columns will be dropped")

# ────────────────────────────────────────────────────────────────────────
# 2) Load base & add tables (exactly как в lgb_training.py)
# ────────────────────────────────────────────────────────────────────────
log("Loading base & add data…")
if folder == 'training':
    train     = pd.read_csv(f'{INPUT_DIR}/train_part.csv')
    train_add = pd.read_csv(f'{INPUT_DIR}/train_part_add.csv')
else:
    train     = pd.read_csv(f'{INPUT_DIR}/train.csv')
    train_add = pd.read_csv(f'{INPUT_DIR}/train_add.csv')
test     = pd.read_csv(f'{INPUT_DIR}/test.csv')
test_add = pd.read_csv(f'{INPUT_DIR}/test_add.csv')

log(f"  train     : {train.shape}   mem={mem(train)}")
log(f"  train_add : {train_add.shape}")
log(f"  test      : {test.shape}")
log(f"  test_add  : {test_add.shape}")

y_train = train.pop('target').values
test_id = test.pop('id').values

# ────────────────────────────────────────────────────────────────────────
# 3) Inject “add” features
# ────────────────────────────────────────────────────────────────────────
log("Injecting add-features…")
cols_add = [
    'msno_artist_name_prob','msno_first_genre_id_prob','msno_xxx_prob',
    'msno_language_prob','msno_yy_prob','source','msno_source_prob',
    'song_source_system_tab_prob','song_source_screen_name_prob',
    'song_source_type_prob'
]
train_add['source'] = train_add['source'].astype('category')
test_add ['source'] = test_add ['source'].astype('category')
for c in cols_add:
    train[c] = train_add[c].values
    test [c] = test_add [c].values

del train_add, test_add; gc.collect()
log(f"  after inject: train has {train.shape[1]} cols   mem={mem(train)}")

# ────────────────────────────────────────────────────────────────────────
# 4) Merge member & member_add
# ────────────────────────────────────────────────────────────────────────
log("Merging member & member_add…")
member     = pd.read_csv(f'{INPUT_DIR}/members_gbdt.csv')
member_add = pd.read_csv(f'{INPUT_DIR}/members_add.csv')

train = train.merge(member, on='msno', how='left')
test  = test .merge(member, on='msno', how='left')
del member; gc.collect()

train = train.merge(member_add[['msno','msno_song_length_mean','artist_msno_cnt']],
                    on='msno', how='left')
test  = test .merge(member_add[['msno','msno_song_length_mean','artist_msno_cnt']],
                    on='msno', how='left')
del member_add; gc.collect()
log(f"  after member merge: train shape = {train.shape}   mem={mem(train)}")

# ────────────────────────────────────────────────────────────────────────
# 5) Merge songs_gbdt (+ before_/after_)
# ────────────────────────────────────────────────────────────────────────
log("Merging songs_gbdt features…")
songs_header = pd.read_csv(f'{INPUT_DIR}/songs_gbdt.csv', nrows=0).columns.tolist()
keep_songs   = [c for c in songs_header if c not in drop_cols]
usecols_songs= ['song_id'] + [c for c in keep_songs if c!='song_id']

songs = pd.read_csv(f'{INPUT_DIR}/songs_gbdt.csv', usecols=usecols_songs)
for c in songs.select_dtypes('float64'): songs[c] = songs[c].astype('float32')
for c in songs.select_dtypes('int64'):   songs[c] = songs[c].astype('int32')

train = train.merge(songs, on='song_id', how='left')
test  = test .merge(songs, on='song_id', how='left')
scols = songs.columns.tolist()
songs.columns = [f'before_{c}' for c in scols]
train = train.merge(songs, on='before_song_id', how='left')
test  = test .merge(songs, on='before_song_id', how='left')
songs.columns = [f'after_{c}' for c in scols]
train = train.merge(songs, on='after_song_id', how='left')
test  = test .merge(songs, on='after_song_id', how='left')
del songs; gc.collect()
log(f"  after songs merge: train shape = {train.shape}   mem={mem(train)}")

# ────────────────────────────────────────────────────────────────────────
# 6) Compute contextual & temporal features
# ────────────────────────────────────────────────────────────────────────
log("Computing extra features…")
for df in (train, test):
    df['before_type_same']   = (df['before_source_type']==df['source_type']).astype('int8')
    df['after_type_same']    = (df['after_source_type']==df['source_type']).astype('int8')
    df['before_artist_same'] = (df['before_artist_name']==df['artist_name']).astype('int8')
    df['after_artist_same']  = (df['after_artist_name']==df['artist_name']).astype('int8')
    df['time_spent']         = (df['timestamp']-df['registration_init_time']).astype('int32')
    df['time_left']          = (df['expiration_date']-df['timestamp']).astype('int32')
    df['duration']           = (df['expiration_date']-df['registration_init_time']).astype('int32')
log(f"  after extras: train mem={mem(train)}")

# ────────────────────────────────────────────────────────────────────────
# 7) Drop low-importance features & free memory
# ────────────────────────────────────────────────────────────────────────
log("Dropping low-importance features…")
train.drop(columns=drop_cols, errors='ignore', inplace=True)
test .drop(columns=drop_cols, errors='ignore', inplace=True)
gc.collect()
log(f"  final train: {train.shape}   mem={mem(train)}")
log(f"  final test : {test.shape}")

# ────────────────────────────────────────────────────────────────────────
# 8) Define models (100 iter/trees each)
# ────────────────────────────────────────────────────────────────────────
models = {
    'dec_tree': DecisionTreeClassifier(min_samples_leaf=3),
    'log_reg' : LogisticRegression(max_iter=100, n_jobs=12, solver='sag'),
    'rf'      : RandomForestClassifier(n_estimators=100, n_jobs=12, min_samples_leaf=3),
    'extra_t' : ExtraTreesClassifier(n_estimators=100, n_jobs=12, min_samples_leaf=3),
    'ada'     : AdaBoostClassifier(n_estimators=100, learning_rate=0.5),
    'gb'      : GradientBoostingClassifier(n_estimators=100, max_depth=3),
    'hist_gb' : HistGradientBoostingClassifier(max_iter=100),
    'knn'     : KNeighborsClassifier(n_neighbors=15, n_jobs=12),
    'lin_svm' : LinearSVC(max_iter=1000),
    'rbf_svm' : SVC(kernel='rbf', probability=True, cache_size=2048),
    'gauss_nb': GaussianNB(),
    'multi_nb': MultinomialNB()
}
if HAVE_XGB:
    models['xgb'] = XGBClassifier(n_estimators=100,
                                   learning_rate=0.3,
                                   max_depth=6,
                                   subsample=0.8,
                                   colsample_bytree=0.8,
                                   n_jobs=8,
                                   use_label_encoder=False,
                                   eval_metric='logloss')

# ────────────────────────────────────────────────────────────────────────
# 9) Train & evaluate
# ────────────────────────────────────────────────────────────────────────
X_train = train.values
del train; gc.collect()

summary = []
total = len(models)                           # сколько всего алгоритмов

for idx, (name, clf) in enumerate(models.items(), start=1):
    log(f"[{idx:02d}/{total:02d}] Training {name} …")
    t0 = time.time()
    try:
        clf.fit(X_train, y_train)

        if hasattr(clf, 'predict_proba'):
            prob = clf.predict_proba(X_train)[:, 1]
        else:                                # SVM-style decision_function
            dfun = clf.decision_function(X_train)
            prob = 1 / (1 + np.exp(-dfun))

    except Exception as e:
        log(f"  ❌  {name} failed: {e}")
        continue

    elapsed = time.time() - t0
    metrics = get_metrics(name, y_train, prob)
    metrics['time_s'] = int(elapsed)
    summary.append(metrics)

    joblib.dump(clf, os.path.join(MODEL_DIR, f"{name}.pkl"))

    log("  " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != 'Model'))
    log(f"  ✅  {name} done in {elapsed:.1f}s\n")

# ────────────────────────────────────────────────────────────────────────
# 10) Summary
# ────────────────────────────────────────────────────────────────────────
res = pd.DataFrame(summary).sort_values('AUC', ascending=False).reset_index(drop=True)
out_path = os.path.join(MODEL_DIR, 'ml_summary.csv')
res.to_csv(out_path, index=False, float_format='%.4f')
print("\n=== SUMMARY ===")
print(res)
