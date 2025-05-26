import os
import numpy as np
import pandas as pd

# загружаем подготовленные данные
train  = pd.read_csv('../temporal_data/train_id_cnt_svd_stamp_before_after.csv')
test   = pd.read_csv('../temporal_data/test_id_cnt_svd_stamp_before_after.csv')
member = pd.read_csv('../temporal_data/members_id_cnt_svd_stamp.csv')
song   = pd.read_csv('../temporal_data/songs_id_cnt_isrc_svd_stamp.csv')

# сохраняем общий train/test в формате для внешних скриптов или проверки
train.to_csv('../train.csv', index=False, float_format='%.6f')
test .to_csv('../test.csv',  index=False, float_format='%.6f')

# помечаем, какие пары (user, song) уже встречаются в тесте, и удаляем их из train
train['iid']     = train['song_id'] * 100000 + train['msno']
test ['iid']     = test ['song_id'] * 100000 + test ['msno']
iid_set          = set(test['iid'].values)
train['appeared'] = train['iid'].isin(iid_set)
train = train.loc[~train['appeared']].drop(['iid','appeared'], axis=1)
train.to_csv('../train_part.csv', index=False, float_format='%.6f')

# сохраняем полные таблицы member и song для GBDT
member.to_csv('../members_gbdt.csv', index=False)

# заполняем пропуски и приводим типы для признаков в song
gbdt_cols = [
    'composer','lyricist','language',
    'first_genre_id','second_genre_id','third_genre_id'
]
for c in gbdt_cols:
    song[c] = song[c].fillna(0).astype(int)
song['artist_name']  = song['artist_name']\
    .fillna(song['artist_name'].max()+1).astype(int)
song['isrc_missing'] = song['isrc_missing'].astype(int)
song.to_csv('../songs_gbdt.csv', index=False)

# формируем данные для нейронной сети: заполняем пропуски и добавляем бинарные флаги
member['bd_missing']             = member['bd'].isna().astype(int)
member['bd']                     = member['bd'].fillna(member['bd'].mean())
member['msno_timestamp_std']     = member['msno_timestamp_std']\
    .fillna(member['msno_timestamp_std'].min())
member.to_csv('../members_nn.csv', index=False)

song['song_id_missing'] = song['song_length'].isna().astype(int)
nn_cols = [
    'song_length','genre_id_cnt','artist_song_cnt','composer_song_cnt',
    'lyricist_song_cnt','genre_song_cnt','song_rec_cnt','artist_rec_cnt',
    'composer_rec_cnt','lyricist_rec_cnt','genre_rec_cnt','yy',
    'cc_song_cnt','xxx_song_cnt','yy_song_cnt','cc_rec_cnt','xxx_rec_cnt',
    'yy_rec_cnt','song_timestamp_std','artist_cnt','lyricist_cnt',
    'composer_cnt','is_featured'
] + [f'artist_component_{i}' for i in range(16)]
for c in nn_cols:
    song[c] = song[c].fillna(song[c].mean())
song.to_csv('../songs_nn.csv', index=False)
