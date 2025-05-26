import numpy as np
import pandas as pd

# читаем предварительно подготовленные данные
train        = pd.read_csv('../temporal_data/train_id.csv')
test         = pd.read_csv('../temporal_data/test_id.csv')
member       = pd.read_csv('../temporal_data/members_id.csv')
song_origin  = pd.read_csv('../temporal_data/songs_id.csv')
song_extra   = pd.read_csv('../temporal_data/songs_extra_id.csv')

# собираем полный список песен по их id
song = pd.DataFrame({
    'song_id': range(max(train.song_id.max(), test.song_id.max()) + 1)
})
song = song.merge(song_origin, on='song_id', how='left')
song = song.merge(song_extra, on='song_id', how='left')

# объединяем пары (msno, song_id) из train и test
data = pd.concat([
    train[['msno', 'song_id']],
    test [['msno', 'song_id']]
], ignore_index=True)

# === считаем, сколько песен у каждого пользователя всего ===
mem_rec_cnt = data.groupby('msno')['song_id'].count().to_dict()
member['msno_rec_cnt'] = member['msno'].map(mem_rec_cnt)

# очищаем возраст: оставляем только от 1 до 74
member['bd'] = member['bd'].apply(lambda x: np.nan if x <= 0 or x >= 75 else x)

# === считаем, сколько песен каждого артиста/композитора/лирического автора ===
artist_song_cnt = song.groupby('artist_name')['song_id'].count().to_dict()
song['artist_song_cnt'] = song['artist_name'].map(artist_song_cnt)

composer_song_cnt = song.groupby('composer')['song_id'].count().to_dict()
composer_song_cnt[0] = np.nan
song['composer_song_cnt'] = song['composer'].map(composer_song_cnt)

lyricist_song_cnt = song.groupby('lyricist')['song_id'].count().to_dict()
lyricist_song_cnt[0] = np.nan
song['lyricist_song_cnt'] = song['lyricist'].map(lyricist_song_cnt)

genre_song_cnt = song.groupby('first_genre_id')['song_id'].count().to_dict()
genre_song_cnt[0] = np.nan
song['genre_song_cnt'] = song['first_genre_id'].map(genre_song_cnt)

# добавляем информацию о пользователе-песнях обратно в data
data = data.merge(song, on='song_id', how='left')

# === считаем, сколько раз каждая песня слушалась ===
song_rec_cnt = data.groupby('song_id')['msno'].count().to_dict()
song['song_rec_cnt'] = song['song_id'].map(song_rec_cnt)

# === сколько раз слушали каждого артиста/композитора/лирического автора ===
artist_rec_cnt = data.groupby('artist_name')['msno'].count().to_dict()
song['artist_rec_cnt'] = song['artist_name'].map(artist_rec_cnt)

composer_rec_cnt = data.groupby('composer')['msno'].count().to_dict()
composer_rec_cnt[0] = np.nan
song['composer_rec_cnt'] = song['composer'].map(composer_rec_cnt)

lyricist_rec_cnt = data.groupby('lyricist')['msno'].count().to_dict()
lyricist_rec_cnt[0] = np.nan
song['lyricist_rec_cnt'] = song['lyricist'].map(lyricist_rec_cnt)

genre_rec_cnt = data.groupby('first_genre_id')['msno'].count().to_dict()
genre_rec_cnt[0] = np.nan
song['genre_rec_cnt'] = song['first_genre_id'].map(genre_rec_cnt)

# === контекстные признаки пользователя (one-hot по source_* → среднее по msno) ===
dummy_feat = ['source_system_tab', 'source_screen_name', 'source_type']
# объединяем train и test без меток
concat = pd.concat([
    train.drop('target', axis=1),
    test .drop('id', axis=1)
], ignore_index=True)

for feat in dummy_feat:
    # one-hot-кодируем
    feat_dummies = pd.get_dummies(concat[feat], prefix=f'msno_{feat}')
    feat_dummies['msno'] = concat['msno']
    # усредняем по пользователю
    feat_stats = feat_dummies.groupby('msno').mean().reset_index()
    # объединяем с таблицей member
    member = member.merge(feat_stats, on='msno', how='left')

# === вытаскиваем вероятности прямо в train/test ===
train_temp = train.merge(member, on='msno', how='left')
test_temp  = test .merge(member, on='msno', how='left')

# для каждого источника делаем msno_source_*_prob
for feat in dummy_feat:
    col_prefix = f'msno_{feat}_'
    # train
    train[f'{col_prefix}prob'] = train_temp.apply(
        lambda row: row[f'{col_prefix}{int(row[feat])}'], axis=1
    )
    # test
    test [f'{col_prefix}prob'] = test_temp.apply(
        lambda row: row[f'{col_prefix}{int(row[feat])}'], axis=1
    )

# === сохраняем результаты ===
# логарифмируем msno_rec_cnt
member['msno_rec_cnt'] = np.log1p(member['msno_rec_cnt'])
member.to_csv('../temporal_data/members_id_cnt.csv', index=False)

# логарифмируем числовые признаки песен
song_feats = [
    'song_length','song_rec_cnt','artist_song_cnt','composer_song_cnt',
    'lyricist_song_cnt','genre_song_cnt','artist_rec_cnt','composer_rec_cnt',
    'lyricist_rec_cnt','genre_rec_cnt'
]
for feat in song_feats:
    song[feat] = np.log1p(song[feat])

song.to_csv('../temporal_data/songs_id_cnt.csv', index=False)

train.to_csv('../temporal_data/train_id_cnt.csv', index=False)
test .to_csv('../temporal_data/test_id_cnt.csv',  index=False)
