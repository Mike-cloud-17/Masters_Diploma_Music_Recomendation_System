import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# загружаем train_part и test, а также характеристики песен для NN
tr   = pd.read_csv('../train_part.csv')
te   = pd.read_csv('../test.csv')
song = pd.read_csv('../songs_nn.csv')

# объединяем нужные столбцы из tr и te
concat = pd.concat([
    tr[['msno','song_id','source_system_tab','source_screen_name','source_type']],
    te[['msno','song_id','source_system_tab','source_screen_name','source_type']]
], ignore_index=True)

# добавляем информацию о песнях по song_id
concat = concat.merge(
    song[[
        'song_id','song_length','artist_name','first_genre_id',
        'artist_rec_cnt','song_rec_cnt','artist_song_cnt','xxx','yy','language'
    ]],
    on='song_id',
    how='left'
)

# формируем единый признак «source» и кодируем его числами
concat['source'] = (
    concat['source_system_tab'] * 10000
  + concat['source_screen_name'] * 100
  + concat['source_type']
)
concat['source'] = LabelEncoder().fit_transform(concat['source'].values)

# ----------------------- вычисление признаков для пользователей -----------------------

# создаём DataFrame с уникальными msno
mem_add = pd.DataFrame({'msno': range(concat['msno'].max() + 1)})

# средние значения ряда признаков по каждому пользователю
data_avg = (
    concat[['msno','song_length','artist_song_cnt','artist_rec_cnt','song_rec_cnt','yy']]
    .groupby('msno')
    .mean()
    .reset_index()   # чтобы msno стал обычным столбцом
)
# переименовываем столбцы
data_avg.columns = ['msno'] + [
    f'msno_{col}_mean' for col in
    ['song_length','artist_song_cnt','artist_rec_cnt','song_rec_cnt','yy']
]
# сливаем с mem_add
mem_add = mem_add.merge(data_avg, on='msno', how='left')

# стандартные отклонения тех же признаков
data_std = (
    concat[['msno','song_length','artist_song_cnt','artist_rec_cnt','song_rec_cnt','yy']]
    .groupby('msno')
    .std()
    .reset_index()
)
data_std.columns = ['msno'] + [
    f'msno_{col}_std' for col in
    ['song_length','artist_song_cnt','artist_rec_cnt','song_rec_cnt','yy']
]
mem_add = mem_add.merge(data_std, on='msno', how='left')

# количество уникальных артистов, слушавшихся пользователем
artist_msno = (
    concat[['msno','artist_name']]
    .groupby('msno')['artist_name']
    .apply(lambda x: len(set(x)))
    .reset_index(name='artist_msno_cnt')
)
artist_msno['artist_msno_cnt'] = np.log1p(artist_msno['artist_msno_cnt'])
mem_add = mem_add.merge(artist_msno, on='msno', how='left')

# вероятности языков: one-hot и среднее по пользователю
lang_dummy = pd.get_dummies(concat['language'], prefix='msno_language')
lang_dummy['msno'] = concat['msno'].values
lang_prob = (
    lang_dummy
    .groupby('msno')
    .mean()
    .reset_index()
)
mem_add = mem_add.merge(lang_prob, on='msno', how='left')

# сохраняем признаки пользователей
mem_add.to_csv('../members_add.csv', index=False)

# ------------------- вычисление признаков для train_part/test -------------------

# список категориальных признаков для подсчёта
features = ['artist_name','first_genre_id','xxx','language','yy','source']

# для каждого признака считаем, сколько раз он встречается у пользователя
for feat in features:
    concat['tmp_id'] = concat['msno'] * 100000 + concat[feat]
    cnt = concat.groupby('tmp_id')['msno'].count().to_dict()
    concat[f'msno_{feat}_cnt'] = concat['tmp_id'].map(cnt)
del concat['tmp_id']

# общее число прослушиваний пользователя
total_msno = concat.groupby('msno')['song_id'].count().to_dict()
concat['msno_cnt'] = concat['msno'].map(total_msno)

# вероятность каждого признака у пользователя
for feat in features:
    concat[f'msno_{feat}_prob'] = (
        concat[f'msno_{feat}_cnt'] / concat['msno_cnt']
    )

# аналогичные счётчики для песен по source_*
for col in ['source_system_tab','source_screen_name','source_type']:
    concat['tmp_id'] = concat['song_id'] * 10000 + concat[col]
    cnt = concat.groupby('tmp_id')['msno'].count().to_dict()
    concat[f'song_{col}_cnt'] = concat['tmp_id'].map(cnt)
del concat['tmp_id']

# общее число прослушиваний каждой песни
total_song = concat.groupby('song_id')['msno'].count().to_dict()
concat['song_cnt'] = concat['song_id'].map(total_song)

# вероятность каждого признака у песни
for col in ['source_system_tab','source_screen_name','source_type']:
    concat[f'song_{col}_prob'] = (
        concat[f'song_{col}_cnt'] / concat['song_cnt']
    )

# выбираем итоговые признаки и делим обратно на tr и te
result_cols = [
    'msno_artist_name_prob','msno_first_genre_id_prob','msno_xxx_prob',
    'msno_language_prob','msno_yy_prob','song_source_system_tab_prob',
    'song_source_screen_name_prob','song_source_type_prob','source',
    'msno_source_prob'
]
result = concat[result_cols]

# сохраняем готовые фичи
result.iloc[:len(tr)].to_csv('../train_part_add.csv', index=False)
result.iloc[len(tr):].to_csv('../test_add.csv',    index=False)
