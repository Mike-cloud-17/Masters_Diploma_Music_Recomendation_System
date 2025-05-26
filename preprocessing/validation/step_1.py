import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

# загружаем исходные данные
members      = pd.read_csv('../source_data/members.csv')
songs        = pd.read_csv('../source_data/songs.csv')
songs_extra  = pd.read_csv('../source_data/song_extra_info.csv')
train        = pd.read_csv('../source_data/train.csv')
test         = pd.read_csv('../source_data/test.csv')

# собираем все song_id из train и test
all_song_ids = pd.concat([train['song_id'], test['song_id']], ignore_index=True)
song_id_set = set(all_song_ids)

# оставляем в songs только те записи, song_id которых есть в train или test
songs['appeared'] = songs['song_id'].apply(lambda x: x in song_id_set)
songs = songs[songs['appeared']]
songs.drop('appeared', axis=1, inplace=True)

# то же для songs_extra
songs_extra['appeared'] = songs_extra['song_id'].apply(lambda x: x in song_id_set)
songs_extra = songs_extra[songs_extra['appeared']]
songs_extra.drop('appeared', axis=1, inplace=True)

# собираем все msno из train и test
all_msnos = pd.concat([train['msno'], test['msno']], ignore_index=True)
msno_set = set(all_msnos)

# оставляем в members только тех пользователей, которые есть в train или test
members['appeared'] = members['msno'].apply(lambda x: x in msno_set)
members = members[members['appeared']]
members.drop('appeared', axis=1, inplace=True)

print('Data loaded.')

# --------------------------------------------------
# кодируем msno
msno_encoder = LabelEncoder()
msno_encoder.fit(members['msno'].values)
members['msno'] = msno_encoder.transform(members['msno'])
train['msno']   = msno_encoder.transform(train['msno'])
test['msno']    = msno_encoder.transform(test['msno'])

print('MSNO done.')

# --------------------------------------------------
# кодируем song_id
song_encoder = LabelEncoder()
all_sids = pd.concat([train['song_id'], test['song_id']], ignore_index=True)
song_encoder.fit(all_sids)
songs['song_id']       = song_encoder.transform(songs['song_id'])
songs_extra['song_id'] = song_encoder.transform(songs_extra['song_id'])
train['song_id']       = song_encoder.transform(train['song_id'])
test['song_id']        = song_encoder.transform(test['song_id'])

print('Song_id done.')

# --------------------------------------------------
# кодируем source_* признаки
for col in ['source_system_tab','source_screen_name','source_type']:
    le = LabelEncoder()
    vals = pd.concat([train[col], test[col]], ignore_index=True)
    le.fit(vals)
    train[col] = le.transform(train[col])
    test[col]  = le.transform(test[col])

print('Source information done.')

# --------------------------------------------------
# кодируем признаки в members
for col in ['city','gender','registered_via']:
    le = LabelEncoder()
    le.fit(members[col])
    members[col] = le.transform(members[col])

# переводим даты в unix timestamp
members['registration_init_time'] = members['registration_init_time']\
    .apply(lambda x: time.mktime(time.strptime(str(x), '%Y%m%d')))
members['expiration_date'] = members['expiration_date']\
    .apply(lambda x: time.mktime(time.strptime(str(x), '%Y%m%d')))

print('Members information done.')

# --------------------------------------------------
# парсим genre_ids в songs
genre_id = np.zeros((len(songs), 4), dtype=int)
for i in range(len(songs)):
    val = songs['genre_ids'].iloc[i]
    # проверяем, что это строка
    if not isinstance(val, str):
        continue
    parts = val.split('|')
    for j in range(min(3, len(parts))):
        genre_id[i, j] = int(parts[j])
    genre_id[i, 3] = len(parts)

songs['first_genre_id']  = genre_id[:, 0]
songs['second_genre_id'] = genre_id[:, 1]
songs['third_genre_id']  = genre_id[:, 2]
songs['genre_id_cnt']    = genre_id[:, 3]

# кодируем все genre_id вместе
gen_encoder = LabelEncoder()
all_genres = pd.concat([
    songs['first_genre_id'],
    songs['second_genre_id'],
    songs['third_genre_id']
], ignore_index=True)
gen_encoder.fit(all_genres)
for col in ['first_genre_id','second_genre_id','third_genre_id']:
    songs[col] = gen_encoder.transform(songs[col])

songs.drop('genre_ids', axis=1, inplace=True)

# --------------------------------------------------
# создаём дополнительные числовые признаки по артистам, композиторам и др.
def artist_count(x):
    return x.count('and') + x.count(',') + x.count(' feat') + x.count('&') + 1

songs['artist_cnt'] = songs['artist_name'].apply(artist_count).astype(np.int8)

def get_count(x):
    try:
        return sum(map(x.count, ['|','/','\\',';'])) + 1
    except:
        return 0

songs['lyricist_cnt'] = songs['lyricist'].apply(get_count).astype(np.int8)
songs['composer_cnt'] = songs['composer'].apply(get_count).astype(np.int8)
songs['is_featured']  = songs['artist_name']\
    .apply(lambda x: 1 if ' feat' in str(x) else 0).astype(np.int8)

# очищаем имена артистов
def get_first_artist(x):
    for sep in ['and', ',', ' feat', '&']:
        if sep in x:
            x = x.split(sep)[0]
    return x.strip()

songs['artist_name'] = songs['artist_name'].apply(get_first_artist)

# очищаем поля lyricist и composer
def get_first_term(x):
    try:
        for sep in ['|','/','\\',';']:
            if sep in x:
                x = x.split(sep)[0]
        return x.strip()
    except:
        return x

songs['lyricist'] = songs['lyricist'].apply(get_first_term)
songs['composer'] = songs['composer'].apply(get_first_term)

# кодируем текстовые признаки
songs['language'] = songs['language'].fillna(-1)
for col in ['artist_name','lyricist','composer','language']:
    le = LabelEncoder()
    le.fit(songs[col])
    songs[col] = le.transform(songs[col])

# --------------------------------------------------
# сохраняем результаты
members.to_csv('../temporal_data/members_id.csv', index=False)
songs.to_csv('../temporal_data/songs_id.csv',    index=False)
songs_extra.to_csv('../temporal_data/songs_extra_id.csv', index=False)
train.to_csv('../temporal_data/train_id.csv',    index=False)
test.to_csv('../temporal_data/test_id.csv',      index=False)
