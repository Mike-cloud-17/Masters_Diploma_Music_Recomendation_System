import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# загружаем train, test и данные по песням для NN
tr   = pd.read_csv('../train_part.csv')
te   = pd.read_csv('../test.csv')
song = pd.read_csv('../songs_nn.csv')

# объединяем необходимые столбцы из tr и te в один датафрейм
concat = pd.concat([
    tr[['msno','song_id','source_system_tab','source_screen_name','source_type']],
    te[['msno','song_id','source_system_tab','source_screen_name','source_type']]
], ignore_index=True)

# добавляем сведения о песнях из song по ключу song_id
concat = concat.merge(
    song[[
        'song_id','song_length','artist_name','first_genre_id',
        'artist_rec_cnt','song_rec_cnt','artist_song_cnt','xxx','yy','language'
    ]],
    on='song_id',
    how='left'
)

# формируем признак «source» как комбинацию трёх кодов
concat['source'] = (
    concat['source_system_tab'] * 10000
  + concat['source_screen_name'] * 100
  + concat['source_type']
)
concat['source'] = LabelEncoder().fit_transform(concat['source'].values)

# --------------------- вычисление признаков для пользователей ---------------------

# создаём пустую таблицу для новых признаков по msno
mem_add = pd.DataFrame({'msno': range(concat['msno'].max() + 1)})

# средние значения ряда признаков по каждому пользователю
data_avg = (
    concat[['msno','song_length','artist_song_cnt','artist_rec_cnt','song_rec_cnt','yy']]
    .groupby('msno')
    .mean()
    .reset_index()   # теперь msno – обычный столбец, а не индекс
)
# переименовываем столбцы, сохраняя msno
data_avg.columns = ['msno'] + ['msno_' + col + '_mean' 
                               for col in ['song_length','artist_song_cnt','artist_rec_cnt','song_rec_cnt','yy']]
# вливаем в mem_add
mem_add = mem_add.merge(data_avg, on='msno', how='left')

# стандартные отклонения тех же признаков
data_std = (
    concat[['msno','song_length','artist_song_cnt','artist_rec_cnt','song_rec_cnt','yy']]
    .groupby('msno')
    .std()
    .reset_index()
)
data_std.columns = ['msno'] + ['msno_' + col + '_std' 
                               for col in ['song_length','artist_song_cnt','artist_rec_cnt','song_rec_cnt','yy']]
mem_add = mem_add.merge(data_std, on='msno', how='left')

# количество уникальных артистов, слушавшихся пользователем
artist_msno = (
    concat[['msno','artist_name']]
    .groupby('msno')['artist_name']
    .apply(lambda x: len(set(x)))
)
mem_add['artist_msno_cnt'] = np.log1p(artist_msno).values

# вероятности языков (one-hot → среднее по пользователю)
language_dummy = pd.get_dummies(concat['language'], prefix='msno_language')
language_dummy['msno'] = concat['msno'].values
language_prob = language_dummy.groupby('msno').mean().reset_index()
mem_add = mem_add.merge(language_prob, on='msno', how='left')

# сохраняем новые признаки для пользователей
mem_add.to_csv('../members_add.csv', index=False)

# ------------------- вычисление признаков для train/test -------------------

# список категориальных признаков для подсчёта
features = ['artist_name','first_genre_id','xxx','language','yy','source']

# для каждого признака считаем, сколько раз он встречается у пользователя
for feat in features:
    concat['id'] = concat['msno'] * 100000 + concat[feat]
    id_cnt = concat.groupby('id')['msno'].count().to_dict()
    concat['msno_' + feat + '_cnt'] = concat['id'].map(id_cnt)

# общее число прослушиваний пользователя
msno_total = concat.groupby('msno')['song_id'].count().to_dict()
concat['msno_cnt'] = concat['msno'].map(msno_total)

# вероятность каждого признака у пользователя
for feat in features:
    concat['msno_' + feat + '_prob'] = (
        concat['msno_' + feat + '_cnt'] / concat['msno_cnt']
    )

# считаем аналогичные счётчики для песен
for col in ['source_system_tab','source_screen_name','source_type']:
    concat['id'] = concat['song_id'] * 10000 + concat[col]
    id_cnt = concat.groupby('id')['msno'].count().to_dict()
    concat['song_' + col + '_cnt'] = concat['id'].map(id_cnt)

# общее число прослушиваний каждой песни
song_total = concat.groupby('song_id')['msno'].count().to_dict()
concat['song_cnt'] = concat['song_id'].map(song_total)

# вероятность каждого признака у песни
for col in ['source_system_tab','source_screen_name','source_type']:
    concat['song_' + col + '_prob'] = (
        concat['song_' + col + '_cnt'] / concat['song_cnt']
    )

# выбираем итоговые признаки и делим обратно на train и test
result_cols = [
    'msno_artist_name_prob','msno_first_genre_id_prob','msno_xxx_prob',
    'msno_language_prob','msno_yy_prob','song_source_system_tab_prob',
    'song_source_screen_name_prob','song_source_type_prob','source',
    'msno_source_prob'
]
result = concat[result_cols]

result.iloc[:len(tr)].to_csv('../train_part_add.csv', index=False)
result.iloc[len(tr):].to_csv('../test_add.csv',    index=False)
