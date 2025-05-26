import numpy as np
import pandas as pd

# загрузка исходных данных
train       = pd.read_csv('../temporal_data/train_id.csv')
test        = pd.read_csv('../temporal_data/test_id.csv')
member      = pd.read_csv('../temporal_data/members_id.csv')
song_origin = pd.read_csv('../temporal_data/songs_id.csv')
song_extra  = pd.read_csv('../temporal_data/songs_extra_id.csv')

# строим таблицу со всеми song_id
song = pd.DataFrame({
    'song_id': range(max(train.song_id.max(), test.song_id.max()) + 1)
})
song = song.merge(song_origin, on='song_id', how='left')
song = song.merge(song_extra,  on='song_id', how='left')

# объединяем пары (msno, song_id) из train и test
data = pd.concat([
    train[['msno', 'song_id']],
    test [['msno', 'song_id']]
], ignore_index=True)

# считаем, сколько разных песен у каждого пользователя
mem_rec_cnt = data.groupby('msno')['song_id'].count().to_dict()
member['msno_rec_cnt'] = member['msno'].map(mem_rec_cnt)

# фильтруем некорректный возраст
member['bd'] = member['bd'].apply(lambda x: np.nan if x <= 0 or x >= 75 else x)

# считаем, сколько песен связано с каждым артистом/композитором/лир.автором/жанром
artist_song_cnt   = song.groupby('artist_name')['song_id'].count().to_dict()
composer_song_cnt = song.groupby('composer'   )['song_id'].count().to_dict()
lyricist_song_cnt = song.groupby('lyricist'  )['song_id'].count().to_dict()
genre_song_cnt    = song.groupby('first_genre_id')['song_id'].count().to_dict()

# значения 0 считаем как отсутствующие
composer_song_cnt[0] = np.nan
lyricist_song_cnt[0] = np.nan
genre_song_cnt   [0] = np.nan

song['artist_song_cnt']   = song['artist_name'].map(artist_song_cnt)
song['composer_song_cnt'] = song['composer'   ].map(composer_song_cnt)
song['lyricist_song_cnt'] = song['lyricist'  ].map(lyricist_song_cnt)
song['genre_song_cnt']    = song['first_genre_id'].map(genre_song_cnt)

# добавляем эти признаки в data по song_id
data = data.merge(song, on='song_id', how='left')

# считаем повторные прослушивания (rec_cnt)
song_rec_cnt    = data.groupby('song_id')      ['msno'].count().to_dict()
artist_rec_cnt  = data.groupby('artist_name')  ['msno'].count().to_dict()
composer_rec_cnt= data.groupby('composer')     ['msno'].count().to_dict()
lyricist_rec_cnt= data.groupby('lyricist')     ['msno'].count().to_dict()
genre_rec_cnt   = data.groupby('first_genre_id')['msno'].count().to_dict()

composer_rec_cnt[0] = np.nan
lyricist_rec_cnt[0] = np.nan
genre_rec_cnt   [0] = np.nan

song['song_rec_cnt']     = song['song_id'      ].map(song_rec_cnt)
song['artist_rec_cnt']   = song['artist_name'  ].map(artist_rec_cnt)
song['composer_rec_cnt'] = song['composer'     ].map(composer_rec_cnt)
song['lyricist_rec_cnt'] = song['lyricist'     ].map(lyricist_rec_cnt)
song['genre_rec_cnt']    = song['first_genre_id'].map(genre_rec_cnt)

# контекстные признаки для msno: one-hot по типу источника
dummy_feat = ['source_system_tab', 'source_screen_name', 'source_type']
concat = pd.concat([
    train.drop('target', axis=1),
    test .drop('id',     axis=1)
], ignore_index=True)

for feat in dummy_feat:
    # создаём dummy-столбцы
    feat_dummies = pd.get_dummies(concat[feat])
    feat_dummies.columns = ['msno_%s_'%feat + str(col) for col in feat_dummies.columns]
    feat_dummies['msno'] = concat['msno'].values
    # усредняем по пользователю
    feat_stats = feat_dummies.groupby('msno').mean().reset_index()
    # добавляем в member
    member = member.merge(feat_stats, on='msno', how='left')

# готовим таблицы для вычисления вероятностей
train_temp = train.merge(member, on='msno', how='left')
test_temp  = test .merge(member, on='msno', how='left')

# для каждого источника считаем P(source | msno)
train['msno_source_system_tab_prob'] = train_temp[[
    col for col in train_temp.columns if 'source_system_tab' in col
]].apply(lambda x: x['msno_source_system_tab_%d'%x['source_system_tab']], axis=1)
test ['msno_source_system_tab_prob'] = test_temp[[
    col for col in test_temp.columns if 'source_system_tab' in col
]].apply(lambda x: x['msno_source_system_tab_%d'%x['source_system_tab']], axis=1)

train['msno_source_screen_name_prob'] = train_temp[[
    col for col in train_temp.columns if 'source_screen_name' in col
]].apply(lambda x: x['msno_source_screen_name_%d'%x['source_screen_name']], axis=1)
test ['msno_source_screen_name_prob'] = test_temp[[
    col for col in test_temp.columns if 'source_screen_name' in col
]].apply(lambda x: x['msno_source_screen_name_%d'%x['source_screen_name']], axis=1)

train['msno_source_type_prob'] = train_temp[[
    col for col in train_temp.columns if 'source_type' in col
]].apply(lambda x: x['msno_source_type_%d'%x['source_type']], axis=1)
test ['msno_source_type_prob'] = test_temp[[
    col for col in test_temp.columns if 'source_type' in col
]].apply(lambda x: x['msno_source_type_%d'%x['source_type']], axis=1)

# сохраняем результаты
member['msno_rec_cnt'] = np.log1p(member['msno_rec_cnt'])
member.to_csv('../temporal_data/members_id_cnt.csv', index=False)

features = [
    'song_length', 'song_rec_cnt', 'artist_song_cnt', 'composer_song_cnt',
    'lyricist_song_cnt', 'genre_song_cnt', 'artist_rec_cnt', 'composer_rec_cnt',
    'lyricist_rec_cnt', 'genre_rec_cnt'
]
for feat in features:
    song[feat] = np.log1p(song[feat])
song.to_csv('../temporal_data/songs_id_cnt.csv', index=False)

train.to_csv('../temporal_data/train_id_cnt.csv', index=False)
test.to_csv('../temporal_data/test_id_cnt.csv', index=False)
