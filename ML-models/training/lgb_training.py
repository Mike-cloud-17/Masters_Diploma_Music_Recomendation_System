import os
import time
import gc
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

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
feat_imp = pd.read_csv('./lgb_feature_importance.csv')
drop_cols = feat_imp.loc[feat_imp.importance < 85, 'name'].tolist()

#####################################################
## 2) Load train/test & “add” tables
#####################################################
print('[1] Loading base & add data…')
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

del train_add, test_add; gc.collect()
print(f'    after inject: train has {train.shape[1]} cols')

#####################################################
## 4) Merge member & member_add
#####################################################
print('[3] Merging member & member_add…')
member     = pd.read_csv(f'{INPUT_DIR}/members_gbdt.csv')
member_add = pd.read_csv(f'{INPUT_DIR}/members_add.csv')

train = train.merge(member,     on='msno', how='left')
test  = test .merge(member,     on='msno', how='left')
del member; gc.collect()

train = train.merge(
    member_add[['msno','msno_song_length_mean','artist_msno_cnt']],
    on='msno', how='left')
test  = test .merge(
    member_add[['msno','msno_song_length_mean','artist_msno_cnt']],
    on='msno', how='left')
del member_add; gc.collect()

#####################################################
## 5) Merge songs_gbdt.csv with only needed cols
#####################################################
print('[4] Merging songs_gbdt features…')
songs_header = pd.read_csv(f'{INPUT_DIR}/songs_gbdt.csv', nrows=0).columns.tolist()
keep_songs   = [c for c in songs_header if c not in drop_cols]
usecols_songs= ['song_id'] + [c for c in keep_songs if c!='song_id']

songs = pd.read_csv(f'{INPUT_DIR}/songs_gbdt.csv', usecols=usecols_songs)
for c in songs.select_dtypes('float64'): songs[c] = songs[c].astype('float32')
for c in songs.select_dtypes('int64'):   songs[c] = songs[c].astype('int32')

train = train.merge(songs, on='song_id', how='left')
test  = test .merge(songs, on='song_id', how='left')
cols_song = songs.columns.tolist()

songs.columns = [f'before_{c}' for c in cols_song]
train = train.merge(songs, on='before_song_id', how='left')
test  = test .merge(songs, on='before_song_id', how='left')

songs.columns = [f'after_{c}' for c in cols_song]
train = train.merge(songs, on='after_song_id', how='left')
test  = test .merge(songs, on='after_song_id', how='left')

del songs; gc.collect()

#####################################################
## 6) Compute contextual & temporal features
#####################################################
print('[5] Computing extra features…')
for df in (train, test):
    df['before_type_same']   = (df['before_source_type']==df['source_type']).astype('float32')
    df['after_type_same']    = (df['after_source_type']==df['source_type']).astype('float32')
    df['before_artist_same'] = (df['before_artist_name']==df['artist_name']).astype('float32')
    df['after_artist_same']  = (df['after_artist_name']==df['artist_name']).astype('float32')
    df['time_spent']         = (df['timestamp']-df['registration_init_time']).astype('int32')
    df['time_left']          = (df['expiration_date']-df['timestamp']).astype('int32')
    df['duration']           = (df['expiration_date']-df['registration_init_time']).astype('int32')

#####################################################
## 7) Drop low-importance features & free memory
#####################################################
print('[6] Dropping low-importance features…')
train.drop(columns=drop_cols, inplace=True, errors='ignore')
test .drop(columns=drop_cols, inplace=True, errors='ignore')
gc.collect()

#####################################################
## 8) Train LightGBM with time-remaining callback
#####################################################
print('[7] Training LightGBM…')
records = pd.read_csv('./lgb_record.csv').sort_values('val_auc', ascending=False)
best    = records.iloc[0]
num_round     = int(best.bst_rnd)
learning_rate = best.lr

params = {
    'boosting_type':           best.type,
    'objective':               'binary',
    'metric':                  ['binary_logloss','auc'],
    'learning_rate':           learning_rate,
    'num_leaves':              int(best.n_leaf),
    'max_depth':               int(best.n_depth),
    'min_data_in_leaf':        int(best.min_data),
    'feature_fraction':        float(best.feature_frac),
    'bagging_fraction':        float(best.bagging_frac),
    'bagging_freq':            int(best.bagging_freq),
    'lambda_l1':               float(best.l1),
    'lambda_l2':               float(best.l2),
    'min_gain_to_split':       float(best.min_gain),
    'min_sum_hessian_in_leaf': float(best.hessian),
    'verbose':                 1,
    'num_threads':             12,
}
print('    params:', params)
print('    num_round:', num_round)

dtrain = lgb.Dataset(train, label=train_y)

start_time = time.time()
def time_callback(env):
    if env.iteration % 100 == 0 and env.iteration > 0:
        elapsed = time.time() - start_time
        remain  = elapsed / env.iteration * (env.end_iteration - env.iteration)
        print(f"    iter={env.iteration}/{env.end_iteration}, elapsed={elapsed:.1f}s, remaining={remain:.1f}s")

callbacks = [lgb.log_evaluation(period=100), time_callback]
gbm = lgb.train(
    params,
    dtrain,
    num_boost_round=num_round,
    valid_sets=[dtrain],
    callbacks=callbacks
)

#####################################################
## 9) Predict, compute metrics & save model
#####################################################
print('[8] Predicting & saving submission…')
pred_train     = gbm.predict(train)
pred_train_bin = (pred_train > 0.5).astype(int)

print(f"    train AUC:       {roc_auc_score(train_y, pred_train):.5f}")
print(f"    train Accuracy:  {accuracy_score(train_y, pred_train_bin):.5f}")
print(f"    train F1-score:  {f1_score(train_y, pred_train_bin):.5f}")
print(f"    train Precision:{precision_score(train_y, pred_train_bin):.5f}")
print(f"    train Recall:   {recall_score(train_y, pred_train_bin):.5f}")

model_path = os.path.join(MODEL_DIR, 'lgb_model.txt')
gbm.save_model(model_path)
print(f"    model saved to {model_path}")

pred_test = gbm.predict(test)
flag      = np.random.randint(0,65536)
out       = pd.DataFrame({'id': test_id, 'target': pred_test})
out.to_csv(f'{SUBMIT_DIR}/lgb_{best.val_auc:.5f}_{flag}.csv.gz',
           index=False, compression='gzip')
print('    submission saved.')
print('Done.')
