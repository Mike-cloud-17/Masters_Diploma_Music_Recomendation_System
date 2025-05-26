import os
import time
import gc
import datetime
import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import joblib

#####################################################
## Settings & Paths
#####################################################
folder     = 'training'
INPUT_DIR  = f'./input/{folder}'
SUBMIT_DIR = './submission'
MODEL_DIR  = './models'
os.makedirs(SUBMIT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

#####################################################
## 1) Read feature-importance to drop low-impact cols
#####################################################
print('[1] Reading feature importance…')
feat_imp = pd.read_csv('./lgb_feature_importance.csv')
drop_cols = feat_imp.loc[feat_imp.importance < 85, 'name'].tolist()
print(f'    Will drop {len(drop_cols)} low-importance features')

#####################################################
## 2) Load train/test & “add” tables
#####################################################
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

#####################################################
## 3) Inject “add” features
#####################################################
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

#####################################################
## 4) Merge member & member_add
#####################################################
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

#####################################################
## 5) Merge songs_gbdt.csv with only needed cols
#####################################################
print('[5] Merging songs_gbdt features…')
songs_header = pd.read_csv(f'{INPUT_DIR}/songs_gbdt.csv', nrows=0).columns.tolist()
keep_songs   = [c for c in songs_header if c not in drop_cols]
usecols_songs= ['song_id'] + [c for c in keep_songs if c!='song_id']

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

#####################################################
## 6) Compute contextual & temporal features
#####################################################
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

#####################################################
## 7) Drop low-importance features & free memory
#####################################################
print('[7] Dropping low-importance features…')
train.drop(columns=drop_cols, inplace=True, errors='ignore')
test .drop(columns=drop_cols, inplace=True, errors='ignore')
gc.collect()
print(f'    dropped {len(drop_cols)} cols → train now {train.shape[1]} cols')

#####################################################
## 7.5) Подготовка данных для MultinomialNB
#####################################################
print('[7.5] Preparing data for MultinomialNB…')

# 1) категориальные → числовые коды (0 зарезервируем под NaN)
for df in (train, test):
    cat_cols = df.select_dtypes('category').columns
    for c in cat_cols:
        # +1, чтобы «реальные» категории начинались с 1, а 0 останется для NaN
        df[c] = df[c].cat.codes.astype('int32') + 1

# 2) object-строки NB не понимает — либо отбросить, либо захешировать.
#    Ниже самый простой вариант: убрать их.
for df in (train, test):
    obj_cols = df.select_dtypes('object').columns
    if len(obj_cols):
        print(f'        dropping {len(obj_cols)} object columns: {list(obj_cols)[:5]}…')
        df.drop(columns=obj_cols, inplace=True)

# 3) заполнить пропуски нулями
train.fillna(0, inplace=True)
test.fillna(0,  inplace=True)

# 4) NB ожидает неотрицательные значения — «подрежем» возможные отрицательные
for df in (train, test):
    num_cols = df.select_dtypes(include=['int32','int64','float32','float64']).columns
    df[num_cols] = df[num_cols].clip(lower=0)

print(f'        ✓ all NaNs filled, categories encoded, negatives clipped')

#####################################################
## 8) Train GaussianNB with logs
#####################################################
print('[8] Training GaussianNB…')
gnb = GaussianNB()
t0 = time.time()
print(f'    start fit @ {datetime.datetime.now().strftime("%H:%M:%S")}')
gnb.fit(train, train_y)
elapsed = time.time() - t0
print(f'    finished fit @ {datetime.datetime.now().strftime("%H:%M:%S")}, duration={elapsed:.1f}s')

#####################################################
## 9) Predict, compute metrics & save model
#####################################################
print('[9] Predicting & evaluating…')
prob_train = gnb.predict_proba(train)[:,1]
bin_train  = (prob_train > 0.5).astype(int)

print(f'    train AUC      = {roc_auc_score(train_y, prob_train):.5f}')
print(f'    train Accuracy = {accuracy_score(train_y, bin_train):.5f}')
print(f'    train F1       = {f1_score(train_y, bin_train):.5f}')
print(f'    train Precision= {precision_score(train_y, bin_train):.5f}')
print(f'    train Recall   = {recall_score(train_y, bin_train):.5f}')

model_path = os.path.join(MODEL_DIR, 'gnb_model.joblib')
joblib.dump(gnb, model_path)
print(f'    model saved to {model_path}')

print('[10] Predicting on test & saving submission…')
prob_test = gnb.predict_proba(test)[:,1]
flag      = np.random.randint(0,65536)
out       = pd.DataFrame({'id': test_id, 'target': prob_test})
fname     = f'{SUBMIT_DIR}/gnb_{roc_auc_score(train_y,prob_train):.5f}_{flag}.csv.gz'
out.to_csv(fname, index=False, compression='gzip')
print(f'    submission saved to {fname}')
print('Done.')
