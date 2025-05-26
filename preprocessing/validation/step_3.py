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

# -------------------- обработка ISRC --------------------

# разбиваем код ISRC на части
isrc = song['isrc']
song['cc']  = isrc.str.slice(0, 2)
song['xxx'] = isrc.str.slice(2, 5)
song['yy']  = isrc.str.slice(5, 7).astype(float).apply(
    lambda x: 2000 + x if x < 18 else 1900 + x
)

# закодируем строковые части числовыми метками
song['cc']  = LabelEncoder().fit_transform(song['cc'])
song['xxx'] = LabelEncoder().fit_transform(song['xxx'])
song['isrc_missing'] = (song['cc'] == 0).astype(float)

# -------------------- подсчёт количества песен по коду --------------------

# сколько песен каждого кода cc встречается
cc_counts = song.groupby('cc')['song_id'].count().to_dict()
cc_counts[0] = None
song['cc_song_cnt'] = song['cc'].map(lambda x: cc_counts.get(x, None))

# аналогично для xxx
xxx_counts = song.groupby('xxx')['song_id'].count().to_dict()
xxx_counts[0] = None
song['xxx_song_cnt'] = song['xxx'].map(lambda x: xxx_counts.get(x, None))

# и для yy (год)
yy_counts = song.groupby('yy')['song_id'].count().to_dict()
yy_counts[0] = None
song['yy_song_cnt'] = song['yy'].map(lambda x: yy_counts.get(x, None))

# -------------------- подсчёт количества прослушиваний --------------------

# объединяем data с song по song_id
data = data.merge(song, on='song_id', how='left')

# сколько раз встречается каждый cc
cc_rec = data.groupby('cc')['msno'].count().to_dict()
cc_rec[0] = None
song['cc_rec_cnt'] = song['cc'].map(lambda x: cc_rec.get(x, None))

# аналогично для xxx
xxx_rec = data.groupby('xxx')['msno'].count().to_dict()
xxx_rec[0] = None
song['xxx_rec_cnt'] = song['xxx'].map(lambda x: xxx_rec.get(x, None))

# и для yy
yy_rec = data.groupby('yy')['msno'].count().to_dict()
yy_rec[0] = None
song['yy_rec_cnt'] = song['yy'].map(lambda x: yy_rec.get(x, None))

# -------------------- сохранение результатов --------------------

# логарифмируем все накопленные счётчики
for feat in [
    'cc_song_cnt','xxx_song_cnt','yy_song_cnt',
    'cc_rec_cnt','xxx_rec_cnt','yy_rec_cnt'
]:
    song[feat] = np.log1p(song[feat].astype(float))

# удаляем ненужные столбцы и сохраняем
song.drop(['name','isrc'], axis=1, inplace=True)
song.to_csv('../temporal_data/songs_id_cnt_isrc.csv', index=False)
