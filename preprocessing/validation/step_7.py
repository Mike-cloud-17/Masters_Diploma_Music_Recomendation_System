import os
import numpy as np
import pandas as pd

## Шаг 1: загрузка предвычисленных признаков
train  = pd.read_csv('../temporal_data/train_id_cnt_svd_stamp_before_after.csv')
test   = pd.read_csv('../temporal_data/test_id_cnt_svd_stamp_before_after.csv')
member = pd.read_csv('../temporal_data/members_id_cnt_svd_stamp.csv')
song   = pd.read_csv('../temporal_data/songs_id_cnt_isrc_svd_stamp.csv')

## Шаг 2: подготовка файлов train.csv и test.csv
train.to_csv('../train.csv', index=False, float_format='%.6f')
test .to_csv('../test.csv',  index=False, float_format='%.6f')

'''
## (опционально) Шаг 3: формирование train_part.csv — отсекаем пары из теста
train['iid']    = train['song_id'] * 100000 + train['msno']
test ['iid']    = test ['song_id'] * 100000 + test ['msno']
iid_set         = set(test['iid'])
train['appeared']= train['iid'].isin(iid_set)
train            = train.loc[~train['appeared']]
train.drop(['iid','appeared'], axis=1, inplace=True)
train.to_csv('../train_part.csv', index=False, float_format='%.6f')
'''

## Шаг 4: формируем данные для GBDT — просто сохраняем member и song в нужных форматах
member.to_csv('../members_gbdt.csv', index=False)

# для каждой из этих колонок заполняем пропуски нулями и приводим к int
gbdt_cols = [
    'composer','lyricist','language',
    'first_genre_id','second_genre_id','third_genre_id'
]
for col in gbdt_cols:
    song[col].fillna(0, inplace=True)
    song[col] = song[col].astype(int)

# artist_name: незнакомые артисты получают новый код
song['artist_name'].fillna(song['artist_name'].max() + 1, inplace=True)
song['artist_name'] = song['artist_name'].astype(int)

# бинарный признак отсутствия ISRC
song['isrc_missing'] = song['isrc_missing'].astype(int)

song.to_csv('../songs_gbdt.csv', index=False)

## Шаг 5: формируем данные для нейросети — добавляем mask- и impute-признаки
# mask-признак для возраста
member['bd_missing'] = member['bd'].isna().astype(int)
# impute возраста средним
member['bd'].fillna(member['bd'].mean(), inplace=True)
# impute стандартного отклонения времени
member['msno_timestamp_std'].fillna(member['msno_timestamp_std'].min(), inplace=True)

member.to_csv('../members_nn.csv', index=False)

# mask-признак для длины песни
song['song_id_missing'] = song['song_length'].isna().astype(int)

# список всех непрерывных колонок для impute
nn_cols = [
    'song_length','genre_id_cnt','artist_song_cnt','composer_song_cnt',
    'lyricist_song_cnt','genre_song_cnt','song_rec_cnt','artist_rec_cnt',
    'composer_rec_cnt','lyricist_rec_cnt','genre_rec_cnt','yy',
    'cc_song_cnt','xxx_song_cnt','yy_song_cnt','cc_rec_cnt','xxx_rec_cnt',
    'yy_rec_cnt','song_timestamp_std','artist_cnt','lyricist_cnt',
    'composer_cnt','is_featured'
] + [f'artist_component_{i}' for i in range(16)]

# заполняем пропуски средними
for col in nn_cols:
    song[col].fillna(song[col].mean(), inplace=True)

song.to_csv('../songs_nn.csv', index=False)
