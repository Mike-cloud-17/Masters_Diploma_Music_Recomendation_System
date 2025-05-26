import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# загружаем данные
train = pd.read_csv('../temporal_data/train_id.csv')
test  = pd.read_csv('../temporal_data/test_id.csv')
song  = pd.read_csv('../temporal_data/songs_id_cnt.csv')

# объединяем пары (msno, song_id) из train и test
data = pd.concat([
    train[['msno', 'song_id']],
    test [['msno', 'song_id']]
], ignore_index=True)

print('Data loaded.')

# обрабатываем поля isrc
isrc = song['isrc']
song['cc']  = isrc.str.slice(0, 2)
song['xxx'] = isrc.str.slice(2, 5)
song['yy']  = isrc.str.slice(5, 7).astype(float)
# восстанавливаем год по коду
song['yy'] = song['yy'].apply(lambda x: 2000 + x if x < 18 else 1900 + x)

# кодируем страны и триады
song['cc']  = LabelEncoder().fit_transform(song['cc'])
song['xxx'] = LabelEncoder().fit_transform(song['xxx'])
# помечаем пропущенные isrc
song['isrc_missing'] = (song['cc'] == 0).astype(float)

# считаем количество песен по каждому коду cc/xxx/yy
song_cc_cnt  = song.groupby('cc')['song_id'].count().to_dict()
song_cc_cnt[0] = None
song['cc_song_cnt']  = song['cc'].apply(lambda x: song_cc_cnt[x]  if not np.isnan(x) else None)

song_xxx_cnt = song.groupby('xxx')['song_id'].count().to_dict()
song_xxx_cnt[0] = None
song['xxx_song_cnt'] = song['xxx'].apply(lambda x: song_xxx_cnt[x] if not np.isnan(x) else None)

song_yy_cnt  = song.groupby('yy')['song_id'].count().to_dict()
song_yy_cnt[0] = None
song['yy_song_cnt']  = song['yy'].apply(lambda x: song_yy_cnt[x]  if not np.isnan(x) else None)

# добавляем эти признаки в общий датафрейм по song_id
data = data.merge(song, on='song_id', how='left')

# считаем повторные прослушивания по cc/xxx/yy
song_cc_cnt  = data.groupby('cc')['msno'].count().to_dict()
song_cc_cnt[0] = None
song['cc_rec_cnt']  = song['cc'].apply(lambda x: song_cc_cnt[x]  if not np.isnan(x) else None)

song_xxx_cnt = data.groupby('xxx')['msno'].count().to_dict()
song_xxx_cnt[0] = None
song['xxx_rec_cnt'] = song['xxx'].apply(lambda x: song_xxx_cnt[x] if not np.isnan(x) else None)

song_yy_cnt  = data.groupby('yy')['msno'].count().to_dict()
song_yy_cnt[0] = None
song['yy_rec_cnt']  = song['yy'].apply(lambda x: song_yy_cnt[x]  if not np.isnan(x) else None)

# логарифмируем все новые счётчики
features = [
    'cc_song_cnt', 'xxx_song_cnt', 'yy_song_cnt',
    'cc_rec_cnt',  'xxx_rec_cnt',  'yy_rec_cnt'
]
for feat in features:
    song[feat] = np.log1p(song[feat])

# удаляем ненужные поля и сохраняем результат
song.drop(['name', 'isrc'], axis=1, inplace=True)
song.to_csv('../temporal_data/songs_id_cnt_isrc.csv', index=False)
