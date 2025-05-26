import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

# Загрузка данных
members = pd.read_csv('../source_data/members.csv')
songs = pd.read_csv('../source_data/songs.csv')
songs_extra = pd.read_csv('../source_data/song_extra_info.csv')
train = pd.read_csv('../source_data/train.csv')
test = pd.read_csv('../source_data/test.csv')

# Объединяем song_id из train и test
song_ids = pd.concat([train['song_id'], test['song_id']], ignore_index=True)
song_id_set = set(song_ids)

# Оставляем только нужные записи в таблицах songs и songs_extra
songs = songs[songs['song_id'].isin(song_id_set)].copy()
songs_extra = songs_extra[songs_extra['song_id'].isin(song_id_set)].copy()

print('Data loaded.')

# Преобразование msno в числовой формат
msno_ids = pd.concat([train['msno'], test['msno']], ignore_index=True)
msno_encoder = LabelEncoder().fit(msno_ids)
members['msno'] = msno_encoder.transform(members['msno'])
train['msno'] = msno_encoder.transform(train['msno'])
test['msno']  = msno_encoder.transform(test['msno'])
print('MSNO done.')

# Преобразование song_id в числовой формат
song_id_encoder = LabelEncoder().fit(song_ids)
songs['song_id']       = song_id_encoder.transform(songs['song_id'])
songs_extra['song_id'] = song_id_encoder.transform(songs_extra['song_id'])
train['song_id']       = song_id_encoder.transform(train['song_id'])
test['song_id']        = song_id_encoder.transform(test['song_id'])
print('Song_id done.')

# Кодирование категориальных признаков в train и test
for col in ['source_system_tab', 'source_screen_name', 'source_type']:
    values = pd.concat([train[col], test[col]], ignore_index=True)
    encoder = LabelEncoder().fit(values)
    train[col] = encoder.transform(train[col])
    test[col]  = encoder.transform(test[col])
print('Source information done.')

# Кодирование категориальных признаков в members
for col in ['city', 'gender', 'registered_via']:
    encoder = LabelEncoder().fit(members[col])
    members[col] = encoder.transform(members[col])

# Преобразование дат в числовой формат timestamp
members['registration_init_time'] = members['registration_init_time'].astype(str).apply(
    lambda x: time.mktime(time.strptime(x, '%Y%m%d'))
)
members['expiration_date'] = members['expiration_date'].astype(str).apply(
    lambda x: time.mktime(time.strptime(x, '%Y%m%d'))
)
print('Members information done.')

# Обработка колонки genre_ids
genre_id = np.zeros((len(songs), 4), dtype=int)
for i in range(len(songs)):
    val = songs['genre_ids'].iat[i]
    if isinstance(val, str):
        parts = val.split('|')
        for j, gid in enumerate(parts[:3]):
            genre_id[i, j] = int(gid)
        genre_id[i, 3] = len(parts)

songs['first_genre_id']  = genre_id[:, 0]
songs['second_genre_id'] = genre_id[:, 1]
songs['third_genre_id']  = genre_id[:, 2]
songs['genre_id_cnt']    = genre_id[:, 3]

# Совместное кодирование всех трёх столбцов жанров
genre_union = pd.concat([
    songs['first_genre_id'],
    songs['second_genre_id'],
    songs['third_genre_id']
], ignore_index=True)
genre_encoder = LabelEncoder().fit(genre_union)
for col in ['first_genre_id', 'second_genre_id', 'third_genre_id']:
    songs[col] = genre_encoder.transform(songs[col])

songs.drop('genre_ids', axis=1, inplace=True)

# Подсчёт количества артистов и композиторов
songs['artist_cnt'] = songs['artist_name'].apply(
    lambda x: x.count('and') + x.count(',') + x.count(' feat') + x.count('&') + 1
).astype(np.int8)

songs['lyricist_cnt'] = songs['lyricist'].apply(
    lambda x: sum(map(x.count, ['|','/','\\',';'])) + 1 if isinstance(x, str) else 0
).astype(np.int8)

songs['composer_cnt'] = songs['composer'].apply(
    lambda x: sum(map(x.count, ['|','/','\\',';'])) + 1 if isinstance(x, str) else 0
).astype(np.int8)

songs['is_featured'] = songs['artist_name'].apply(
    lambda x: 1 if ' feat' in str(x) else 0
).astype(np.int8)

# Выделение первого имени артиста, композитора и лириста
songs['artist_name'] = songs['artist_name'].apply(
    lambda x: x.split('and')[0].split(',')[0].split(' feat')[0].split('&')[0].strip()
)
songs['lyricist'] = songs['lyricist'].apply(
    lambda x: x.split('|')[0].split('/')[0].split('\\')[0].split(';')[0].strip() if isinstance(x, str) else x
)
songs['composer'] = songs['composer'].apply(
    lambda x: x.split('|')[0].split('/')[0].split('\\')[0].split(';')[0].strip() if isinstance(x, str) else x
)

# Кодирование оставшихся признаков
songs['language'] = songs['language'].fillna(-1)
for col in ['artist_name','lyricist','composer','language']:
    encoder = LabelEncoder().fit(songs[col])
    songs[col] = encoder.transform(songs[col])

# Сохранение результатов
members.to_csv('../temporal_data/members_id.csv', index=False)
songs.to_csv('../temporal_data/songs_id.csv', index=False)
songs_extra.to_csv('../temporal_data/songs_extra_id.csv', index=False)
train.to_csv('../temporal_data/train_id.csv', index=False)
test.to_csv('../temporal_data/test_id.csv', index=False)

print('Step 1 (id_process.py) completed.')
