import os
import gc
import datetime
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

#####################################################
## Settings & Paths
#####################################################
folder     = 'training'
INPUT_DIR  = f'./input/{folder}'
SUBMIT_DIR = './submission'
os.makedirs(SUBMIT_DIR, exist_ok=True)

#####################################################
## 1) Read feature-importance to know which cols to drop
#####################################################
feat_imp = pd.read_csv('./lgb_feature_importance.csv')
drop_cols = feat_imp.loc[feat_imp.importance < 85, 'name'].tolist()

#####################################################
## 2) Load train/test & “add” tables
#####################################################
print('[1] Loading base & add data…')
if folder == 'training':
    train      = pd.read_csv(f'{INPUT_DIR}/train_part.csv')
    train_add  = pd.read_csv(f'{INPUT_DIR}/train_part_add.csv')
else:
    train      = pd.read_csv(f'{INPUT_DIR}/train.csv')
    train_add  = pd.read_csv(f'{INPUT_DIR}/train_add.csv')

test     = pd.read_csv(f'{INPUT_DIR}/test.csv')
test_add = pd.read_csv(f'{INPUT_DIR}/test_add.csv')

print(f'    train:     {train.shape}')
print(f'    train_add: {train_add.shape}')
print(f'    test:      {test.shape}')
print(f'    test_add:  {test_add.shape}')

train_y = train.pop('target')
test_id = test.pop('id')

#####################################################
## 3) Inject “add” features
#####################################################
print('[2] Injecting add-features…')
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

del train_add, test_add
gc.collect()

print(f'    after inject: train has {train.shape[1]} cols')

#####################################################
## 4) Merge member & member_add
#####################################################
print('[3] Merging member & member_add…')
member     = pd.read_csv(f'{INPUT_DIR}/members_gbdt.csv')
member_add = pd.read_csv(f'{INPUT_DIR}/members_add.csv')

train = train.merge(member, on='msno', how='left')
test  = test .merge(member, on='msno', how='left')
print(f'    after member: train {train.shape}, test {test.shape}')
del member; gc.collect()

train = train.merge(member_add[['msno','msno_song_length_mean','artist_msno_cnt']],
                    on='msno', how='left')
test  = test .merge(member_add[['msno','msno_song_length_mean','artist_msno_cnt']],
                    on='msno', how='left')
print(f'    after member_add: train {train.shape}, test {test.shape}')
del member_add; gc.collect()

#####################################################
## 5) Merge songs_gbdt.csv with only needed cols
#####################################################
print('[4] Merging songs_gbdt features…')
# read header to pick usecols
songs_header = pd.read_csv(f'{INPUT_DIR}/songs_gbdt.csv', nrows=0).columns.tolist()
# keep song_id plus those not in drop_cols
keep_songs = [c for c in songs_header if c not in drop_cols]
# ensure song_id is first
usecols_songs = ['song_id'] + [c for c in keep_songs if c != 'song_id']

songs = pd.read_csv(f'{INPUT_DIR}/songs_gbdt.csv', usecols=usecols_songs)

# cast to save memory
for c in songs.select_dtypes('float64').columns:
    songs[c] = songs[c].astype('float32')
for c in songs.select_dtypes('int64').columns:
    songs[c] = songs[c].astype('int32')

# base merge
train = train.merge(songs, on='song_id', how='left')
test  = test .merge(songs, on='song_id', how='left')
print(f'    after base songs: train {train.shape}, test {test.shape}')

# before/after merges
cols_song = songs.columns.tolist()
songs.columns = [f'before_{c}' for c in cols_song]
train = train.merge(songs, on='before_song_id', how='left')
test  = test .merge(songs, on='before_song_id', how='left')
print(f'    after before_song: train {train.shape}, test {test.shape}')

songs.columns = [f'after_{c}' for c in cols_song]
train = train.merge(songs, on='after_song_id', how='left')
test  = test .merge(songs, on='after_song_id', how='left')
print(f'    after after_song: train {train.shape}, test {test.shape}')

del songs
gc.collect()

#####################################################
## 6) Extra contextual & temporal features
#####################################################
print('[5] Computing contextual & temporal features…')
for df in (train, test):
    df['before_type_same']   = (df['before_source_type']   == df['source_type']).astype('float32')
    df['after_type_same']    = (df['after_source_type']    == df['source_type']).astype('float32')
    df['before_artist_same'] = (df['before_artist_name']   == df['artist_name']).astype('float32')
    df['after_artist_same']  = (df['after_artist_name']    == df['artist_name']).astype('float32')
    df['time_spent']         = (df['timestamp']            - df['registration_init_time']).astype('int32')
    df['time_left']          = (df['expiration_date']      - df['timestamp']).astype('int32')
    df['duration']           = (df['expiration_date']      - df['registration_init_time']).astype('int32')

print(f'    after extras: train {train.shape}, test {test.shape}')

#####################################################
## 7) Drop low-importance features & free memory
#####################################################
print('[6] Dropping low-importance features…')
train.drop(columns=drop_cols, inplace=True, errors='ignore')
test .drop(columns=drop_cols, inplace=True, errors='ignore')
print(f'    dropped {len(drop_cols)} cols → train now {train.shape[1]} cols')
gc.collect()

#####################################################
## 8) Train CatBoost with reduced footprint
#####################################################
print('[7] Training CatBoostClassifier…')
rec = pd.read_csv('./lgb_record.csv').sort_values('val_auc', ascending=False).iloc[0]
iters = int(rec.bst_rnd)
lr    = rec.lr
depth = int(rec.n_depth)
l2    = rec.l2

print(f'    params: iterations={iters}, lr={lr:.4f}, depth={depth}, l2_leaf_reg={l2:.2e}')
model = CatBoostClassifier(
    iterations    = 100, # обновить на iters
    learning_rate = lr,
    depth         = depth,
    l2_leaf_reg   = l2,
    border_count  = 31,    # reduced
    thread_count  = 8,     # reduced
    verbose       = 5,
    random_seed   = 42
)

cat_features = ['source','source_system_tab','source_screen_name','source_type']
model.fit(train, train_y, cat_features=[c for c in cat_features if c in train.columns])

#####################################################
## 9) Predict & save
#####################################################
print('[8] Predicting & saving submission…')
pred_train = model.predict_proba(train)[:,1]
print(f'    train AUC: {roc_auc_score(train_y, pred_train):.5f}')

pred_test = model.predict_proba(test)[:,1]
flag      = np.random.randint(0, 65536)
fname     = f'{SUBMIT_DIR}/catboost_{roc_auc_score(train_y,pred_train):.5f}_{flag}.csv.gz'
pd.DataFrame({'id': test_id, 'target': pred_test})\
  .to_csv(fname, index=False, compression='gzip')
print('    saved to', fname)

print('Done.')
