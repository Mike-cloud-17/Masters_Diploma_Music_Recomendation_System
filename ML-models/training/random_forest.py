import os, time, gc, datetime, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score
)
import joblib

# ─────────────────────────────────────────────────────────
# 0) Settings & paths
# ─────────────────────────────────────────────────────────
folder     = 'training'
INPUT_DIR  = f'./input/{folder}'
SUBMIT_DIR = './submission'
MODEL_DIR  = './models'
TMP_DIR    = './tmp'
os.makedirs(SUBMIT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TMP_DIR,     exist_ok=True)

# ─────────────────────────────────────────────────────────
# 1) Read feature importance
# ─────────────────────────────────────────────────────────
print('[1] Reading feature importance…')
feat_imp  = pd.read_csv('./lgb_feature_importance.csv')
drop_cols = feat_imp.loc[feat_imp.importance < 85, 'name'].tolist()
print(f'    dropping {len(drop_cols)} low-importance features')

# ─────────────────────────────────────────────────────────
# 2) Load base & “add” tables
# ─────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────
# 3) Inject “add” features
# ─────────────────────────────────────────────────────────
print('[3] Injecting add-features…')
cols_add = [
    'msno_artist_name_prob','msno_first_genre_id_prob','msno_xxx_prob',
    'msno_language_prob','msno_yy_prob','source','msno_source_prob',
    'song_source_system_tab_prob','song_source_screen_name_prob',
    'song_source_type_prob'
]
train_add['source'] = train_add['source'].astype('category')
test_add ['source'] = test_add['source'].astype('category')
for c in cols_add:
    train[c] = train_add[c].values
    test [c] = test_add [c].values
del train_add, test_add; gc.collect()
print(f'    after inject: train has {train.shape[1]} cols')

# ─────────────────────────────────────────────────────────
# 4) Merge member & member_add
# ─────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────
# 5) Merge songs_gbdt
# ─────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────
# 6) Extra contextual features
# ─────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────
# 7) Drop low-importance cols
# ─────────────────────────────────────────────────────────
print('[7] Dropping low-importance features…')
train.drop(columns=drop_cols, inplace=True, errors='ignore')
test .drop(columns=drop_cols, inplace=True, errors='ignore')
gc.collect()
print(f'    dropped {len(drop_cols)} cols → train now {train.shape[1]} cols')

# ─────────────────────────────────────────────────────────
# 8) Fit RandomForest
# ─────────────────────────────────────────────────────────
print('[8] Training RandomForestClassifier…')
rf_clf = RandomForestClassifier(
    n_estimators     = 100,
    min_samples_leaf = 3,
    n_jobs           = 8,   # многопоточное обучение
    oob_score        = True,
    verbose          = 1
)
t0 = time.time()
print(f'    start fit @ {datetime.datetime.now().strftime("%H:%M:%S")}')
rf_clf.fit(train, train_y)
print(f'    finished fit @ {datetime.datetime.now().strftime("%H:%M:%S")}, '
      f'duration={time.time()-t0:.1f}s')

# ─────────────────────────────────────────────────────────
# 8.1) Off-load test to disk ⟶ освободим RAM
# ─────────────────────────────────────────────────────────
print('[8.1] Off-loading test to disk before train-metrics…')
TEST_FEATHER     = os.path.join(TMP_DIR, 'test.feather')
TEST_ID_FEATHER  = os.path.join(TMP_DIR, 'test_id.feather')
test.reset_index(drop=True).to_feather(TEST_FEATHER)
pd.DataFrame({'id': test_id}).to_feather(TEST_ID_FEATHER)
del test, test_id; gc.collect()
print('        test tables saved & removed from RAM')

# ─────────────────────────────────────────────────────────
# 9) Train-metrics (батч-инференс, памяти хватит)
# ─────────────────────────────────────────────────────────
print('[9] Predicting & evaluating on train…')

def predict_in_batches(model, X, batch=250_000):
    proba = np.empty(X.shape[0], dtype='float32')
    for start in range(0, X.shape[0], batch):
        end = min(start+batch, X.shape[0])
        proba[start:end] = model.predict_proba(X.iloc[start:end])[:, 1]
    return proba

pred_train = predict_in_batches(rf_clf, train)
bin_train  = (pred_train > 0.5).astype(int)

print(f'    train AUC      = {roc_auc_score(train_y, pred_train):.5f}')
print(f'    train Accuracy = {accuracy_score(train_y, bin_train):.5f}')
print(f'    train F1       = {f1_score(train_y, bin_train):.5f}')
print(f'    train Precision= {precision_score(train_y, bin_train):.5f}')
print(f'    train Recall   = {recall_score(train_y, bin_train):.5f}')

# ── save model
model_path = os.path.join(MODEL_DIR, 'rf_model.joblib')
joblib.dump(rf_clf, model_path)
print(f'    model saved to {model_path}')

# ── free train RAM
del train, train_y, pred_train, bin_train; gc.collect()

# ─────────────────────────────────────────────────────────
# 10) Load test back & predict (батчи, 1 поток)
# ─────────────────────────────────────────────────────────
print('[10] Loading test back & predicting for submission…')
test    = pd.read_feather(TEST_FEATHER)
test_id = pd.read_feather(TEST_ID_FEATHER)['id']
os.remove(TEST_FEATHER); os.remove(TEST_ID_FEATHER)

rf_clf.set_params(n_jobs=1)  # экономим память на инференсе
pred_test = predict_in_batches(rf_clf, test, batch=250_000)

flag  = np.random.randint(0, 65536)
subm  = pd.DataFrame({'id': test_id, 'target': pred_test})
fname = f'{SUBMIT_DIR}/rf_{flag}.csv.gz'
subm.to_csv(fname, index=False, compression='gzip')
print(f'    submission saved to {fname}')
print('Done.')
