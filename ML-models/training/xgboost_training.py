import os, time, gc, datetime, numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import joblib

# ────────────────────────────────
# 0) Paths & dirs
# ────────────────────────────────
folder     = 'training'
INPUT_DIR  = f'./input/{folder}'
SUBMIT_DIR = './submission'
MODEL_DIR  = './models'
TMP_DIR    = './tmp'
os.makedirs(SUBMIT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TMP_DIR,     exist_ok=True)

# ────────────────────────────────
# 1) Feature importance → drop list
# ────────────────────────────────
print('[1] Reading feature importance…')
feat_imp  = pd.read_csv('./lgb_feature_importance.csv')
drop_cols = feat_imp.loc[feat_imp.importance < 85, 'name'].tolist()
print(f'    dropping {len(drop_cols)} features')

# ────────────────────────────────
# 2) Load base + add
# ────────────────────────────────
print('[2] Loading base & add data…')
if folder == 'training':
    train     = pd.read_csv(f'{INPUT_DIR}/train_part.csv')
    train_add = pd.read_csv(f'{INPUT_DIR}/train_part_add.csv')
else:
    train     = pd.read_csv(f'{INPUT_DIR}/train.csv')
    train_add = pd.read_csv(f'{INPUT_DIR}/train_add.csv')

test     = pd.read_csv(f'{INPUT_DIR}/test.csv')
test_add = pd.read_csv(f'{INPUT_DIR}/test_add.csv')
print(f'    train:     {train.shape}')
print(f'    train_add: {train_add.shape}')
print(f'    test:      {test.shape}')
print(f'    test_add:  {test_add.shape}')

train_y = train.pop('target')
test_id = test.pop('id')

# ────────────────────────────────
# 3) Inject add-features
# ────────────────────────────────
print('[3] Injecting add-features…')
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
print(f'    after inject: train has {train.shape[1]} cols')

# ────────────────────────────────
# 4) Merge member & member_add
# ────────────────────────────────
print('[4] Merging member & member_add…')
member     = pd.read_csv(f'{INPUT_DIR}/members_gbdt.csv')
member_add = pd.read_csv(f'{INPUT_DIR}/members_add.csv')

train = train.merge(member, on='msno', how='left')
test  = test .merge(member, on='msno', how='left')
del member; gc.collect()

train = train.merge(
    member_add[['msno','msno_song_length_mean','artist_msno_cnt']],
    on='msno', how='left')
test  = test .merge(
    member_add[['msno','msno_song_length_mean','artist_msno_cnt']],
    on='msno', how='left')
del member_add; gc.collect()
print(f'    merged members → train {train.shape}, test {test.shape}')

# ────────────────────────────────
# 5) Merge songs_gbdt
# ────────────────────────────────
print('[5] Merging songs_gbdt features…')
songs_header  = pd.read_csv(f'{INPUT_DIR}/songs_gbdt.csv', nrows=0).columns
keep_songs    = [c for c in songs_header if c not in drop_cols]
usecols_songs = ['song_id'] + [c for c in keep_songs if c != 'song_id']

songs = pd.read_csv(f'{INPUT_DIR}/songs_gbdt.csv', usecols=usecols_songs)
for c in songs.select_dtypes('float64'): songs[c] = songs[c].astype('float32')
for c in songs.select_dtypes('int64'):   songs[c] = songs[c].astype('int32')

train = train.merge(songs, on='song_id', how='left')
test  = test .merge(songs, on='song_id', how='left')
print(f'    after base songs: train {train.shape}, test {test.shape}')

cols_song = songs.columns.tolist()
songs.columns = [f'before_{c}' for c in cols_song]
train = train.merge(songs, on='before_song_id', how='left')
test  = test .merge(songs, on='before_song_id', how='left')
print(f'    after before_song: train {train.shape}, test {test.shape}')

songs.columns = [f'after_{c}' for c in cols_song]
train = train.merge(songs, on='after_song_id', how='left')
test  = test .merge(songs, on='after_song_id', how='left')
print(f'    after after_song: train {train.shape}, test {test.shape}')
del songs; gc.collect()

# ────────────────────────────────
# 6) Extra contextual features
# ────────────────────────────────
print('[6] Computing extra features…')
for df in (train, test):
    df['before_type_same']   = (df['before_source_type']==df['source_type']).astype('float32')
    df['after_type_same']    = (df['after_source_type']==df['source_type']).astype('float32')
    df['before_artist_same'] = (df['before_artist_name']==df['artist_name']).astype('float32')
    df['after_artist_same']  = (df['after_artist_name']==df['artist_name']).astype('float32')
    df['time_spent']         = (df['timestamp']-df['registration_init_time']).astype('int32')
    df['time_left']          = (df['expiration_date']-df['timestamp']).astype('int32')
    df['duration']           = (df['expiration_date']-df['registration_init_time']).astype('int32')
print(f'    after extras: train {train.shape}, test {test.shape}')

# ────────────────────────────────
# 7) Drop low-importance cols
# ────────────────────────────────
print('[7] Dropping low-importance features…')
train.drop(columns=drop_cols, inplace=True, errors='ignore')
test .drop(columns=drop_cols, inplace=True, errors='ignore')
gc.collect()
print(f'    dropped {len(drop_cols)} cols → train now {train.shape[1]} cols')

# ────────────────────────────────
# 7.5) Prepare for XGBoost
# ────────────────────────────────
print('[7.5] Preparing data for XGBoost…')
for df in (train, test):
    # categories → int codes (0 зарезервирован для NaN)
    for c in df.select_dtypes('category').columns:
        df[c] = df[c].cat.codes.astype('int32') + 1
    # убрать object-строки
    obj_cols = df.select_dtypes('object').columns
    if len(obj_cols):
        print(f'        dropping {len(obj_cols)} object cols: {list(obj_cols)[:5]}…')
        df.drop(columns=obj_cols, inplace=True)
    df.fillna(0, inplace=True)
    num_cols = df.select_dtypes(include=['int32','int64','float32','float64']).columns
    df[num_cols] = df[num_cols].clip(lower=0)
gc.collect()
print('        ✓ categories encoded, NaNs filled, negatives clipped')

# ────────────────────────────────
# 8) Train/val split (5 % hold-out)
# ────────────────────────────────
VAL_FRAC = 0.05
val_idx  = train.sample(frac=VAL_FRAC, random_state=42).index
X_val, y_val = train.loc[val_idx],  train_y.loc[val_idx]
X_trn, y_trn = train.drop(index=val_idx), train_y.drop(index=val_idx)
print(f'[8] Training XGBClassifier…   (val set = {len(val_idx):,} rows)')

xgb_clf = XGBClassifier(
    n_estimators        = 1500,     # ← фиксированное число раундов
    learning_rate       = 0.05,
    max_depth           = 8,
    min_child_weight    = 1,
    gamma               = 0.10,
    subsample           = 0.80,
    colsample_bytree    = 0.80,
    reg_lambda          = 1.0,
    tree_method         = 'hist',
    max_bin             = 256,
    eval_metric         = 'auc',
    n_jobs              = 8,
    verbosity           = 0
)

t0 = time.time()
print(f'    start fit @ {datetime.datetime.now().strftime("%H:%M:%S")}')

xgb_clf.fit(
    X_trn, y_trn,
    eval_set=[(X_val, y_val)],
    verbose=50           # ← AUC выводится каждые 50 итераций
)

print(f'    finished fit @ {datetime.datetime.now().strftime("%H:%M:%S")}, '
      f'duration={time.time()-t0:.1f}s')

# ────────────────────────────────
# 8.1) Краткая история AUC (шаг 50)
# ────────────────────────────────
auc_hist = xgb_clf.evals_result()['validation_0']['auc']
print('\n[8.1] AUC history (every 50):')
for i in range(0, len(auc_hist), 50):
    print(f'    iter {i:04d}: AUC={auc_hist[i]:.5f}')
if len(auc_hist) % 50:
    i = len(auc_hist) - 1
    print(f'    iter {i:04d}: AUC={auc_hist[i]:.5f}')

# ────────────────────────────────
# 9) Final validation metrics
# ────────────────────────────────
print('\n[9] Final validation metrics…')
pred_val     = xgb_clf.predict_proba(X_val)[:, 1]
pred_val_bin = (pred_val > 0.5).astype(int)
print(f'    val AUC      = {roc_auc_score(y_val, pred_val):.5f}')
print(f'    val Accuracy = {accuracy_score(y_val, pred_val_bin):.5f}')
print(f'    val F1       = {f1_score(y_val, pred_val_bin):.5f}')
print(f'    val Precision= {precision_score(y_val, pred_val_bin):.5f}')
print(f'    val Recall   = {recall_score(y_val, pred_val_bin):.5f}')

model_path = os.path.join(MODEL_DIR, 'xgb_model.joblib')
joblib.dump(xgb_clf, model_path)
print(f'    model saved to {model_path}')

# ────────────────────────────────
# 10) Free train & predict on test
# ────────────────────────────────
del X_trn, y_trn, X_val, y_val, train, train_y; gc.collect()

print('\n[10] Predicting on test & saving submission…')
pred_test = xgb_clf.predict_proba(test)[:, 1]
flag      = np.random.randint(0, 65536)
subm      = pd.DataFrame({'id': test_id, 'target': pred_test})
fname     = f'{SUBMIT_DIR}/xgb_{flag}.csv.gz'
subm.to_csv(fname, index=False, compression='gzip')
print(f'    submission saved to {fname}')
print('✅ Done.')
