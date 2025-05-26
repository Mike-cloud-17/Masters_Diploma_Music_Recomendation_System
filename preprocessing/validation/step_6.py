import numpy as np
import pandas as pd
from collections import defaultdict

# 1) загружаем данные, полученные после SVD + штампования
tr = pd.read_csv('../temporal_data/train_id_cnt_svd_stamp.csv')
te = pd.read_csv('../temporal_data/test_id_cnt_svd_stamp.csv')

print('Данные загружены:')
print(f'  train: {len(tr)} строк')
print(f'  test : {len(te)} строк')

# 2) объединяем train и test для сквозной обработки
concat = pd.concat([
    tr[['msno','song_id','source_type','source_screen_name','timestamp']],
    te[['msno','song_id','source_type','source_screen_name','timestamp']]
], ignore_index=True)

# 3) вычисляем признаки «до события» для каждого пользователя
song_dict = defaultdict(lambda: None)
type_dict = defaultdict(lambda: None)
name_dict = defaultdict(lambda: None)
time_dict = defaultdict(lambda: None)

before_data = np.zeros((len(concat), 4))
for i in range(len(concat)):
    u = concat.at[i, 'msno']
    if song_dict[u] is None:
        # если ранее не было прослушиваний, берём текущее, но сбрасываем время
        row = concat.loc[i, ['song_id','source_type','source_screen_name','timestamp']].values
        before_data[i] = row
        before_data[i, 3] = np.nan
    else:
        # иначе — прошлое событие пользователя
        before_data[i] = [
            song_dict[u],
            type_dict[u],
            name_dict[u],
            time_dict[u]
        ]
    # обновляем последнее событие
    song_dict[u] = concat.at[i, 'song_id']
    type_dict[u] = concat.at[i, 'source_type']
    name_dict[u] = concat.at[i, 'source_screen_name']
    time_dict[u] = concat.at[i, 'timestamp']

print('Признаки «до события» готовы')

# 4) вычисляем признаки «после события» для каждого пользователя
song_dict.clear(); type_dict.clear(); name_dict.clear(); time_dict.clear()

after_data = np.zeros((len(concat), 4))
for i in range(len(concat)-1, -1, -1):
    u = concat.at[i, 'msno']
    if song_dict[u] is None:
        # если нет последующих прослушиваний, берём текущее, но сбрасываем время
        row = concat.loc[i, ['song_id','source_type','source_screen_name','timestamp']].values
        after_data[i] = row
        after_data[i, 3] = np.nan
    else:
        # иначе — следующее событие пользователя
        after_data[i] = [
            song_dict[u],
            type_dict[u],
            name_dict[u],
            time_dict[u]
        ]
    # обновляем следующее событие
    song_dict[u] = concat.at[i, 'song_id']
    type_dict[u] = concat.at[i, 'source_type']
    name_dict[u] = concat.at[i, 'source_screen_name']
    time_dict[u] = concat.at[i, 'timestamp']

print('Признаки «после события» готовы')

# 5) добавляем полученные признаки обратно в tr и te
cols = ['song_id','source_type','source_screen_name','timestamp']
for idx, col in enumerate(cols):
    tr[f'before_{col}'] = before_data[:len(tr), idx]
    tr[f'after_{col}']  = after_data[:len(tr), idx]
    te[f'before_{col}'] = before_data[len(tr):, idx]
    te[f'after_{col}']  = after_data[len(tr):, idx]

# приводим категориальные к int
for col in ['song_id','source_type','source_screen_name']:
    tr[f'before_{col}'] = tr[f'before_{col}'].astype(int)
    tr[f'after_{col}']  = tr[f'after_{col}'].astype(int)
    te[f'before_{col}'] = te[f'before_{col}'].astype(int)
    te[f'after_{col}']  = te[f'after_{col}'].astype(int)

# 6) вычисляем логарифм разницы времени до/после и заполняем NaN средним
tr['before_timestamp'] = np.log1p(tr['timestamp'] - tr['before_timestamp'])
te['before_timestamp'] = np.log1p(te['timestamp'] - te['before_timestamp'])
tr['after_timestamp']  = np.log1p(tr['after_timestamp'] - tr['timestamp'])
te['after_timestamp']  = np.log1p(te['after_timestamp'] - te['timestamp'])

mean_b = tr['before_timestamp'].mean()
mean_a = tr['after_timestamp'].mean()
tr['before_timestamp'].fillna(mean_b, inplace=True)
te['before_timestamp'].fillna(mean_b, inplace=True)
tr['after_timestamp'].fillna(mean_a, inplace=True)
te['after_timestamp'].fillna(mean_a, inplace=True)

# 7) сохраняем результаты
tr.to_csv('../temporal_data/train_id_cnt_svd_stamp_before_after.csv', index=False)
te.to_csv('../temporal_data/test_id_cnt_svd_stamp_before_after.csv', index=False)

print('Файл before_after_process успешно сохранён')
