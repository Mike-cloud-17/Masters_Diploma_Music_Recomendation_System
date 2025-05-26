import numpy as np
import pandas as pd
from collections import defaultdict

# загружаем данные train и test с временными метками
tr = pd.read_csv('../temporal_data/train_id_cnt_svd_stamp.csv')
te = pd.read_csv('../temporal_data/test_id_cnt_svd_stamp.csv')

print('данные загружены')
print(f'тренировочных строк: {len(tr)}')
print(f'тестовых строк: {len(te)}')

# объединяем train и test для сквозного индексирования событий
concat = pd.concat([
    tr[['msno', 'song_id', 'source_type', 'source_screen_name', 'timestamp']],
    te[['msno', 'song_id', 'source_type', 'source_screen_name', 'timestamp']]
], ignore_index=True)

# готовим структуры для хранения «предыдущих» значений
song_dict = defaultdict(lambda: None)
type_dict = defaultdict(lambda: None)
name_dict = defaultdict(lambda: None)
time_dict = defaultdict(lambda: None)

# рассчитываем признаки «до события»
before_data = np.zeros((len(concat), 4))
for i in range(len(concat)):
    u = concat.at[i, 'msno']
    if song_dict[u] is None:
        # если пользователь ещё не слушал ничего раньше, берём текущее и обнуляем время
        row = concat.loc[i, ['song_id', 'source_type', 'source_screen_name', 'timestamp']].values
        before_data[i] = row
        before_data[i, 3] = np.nan
    else:
        # иначе — предыдущие сохранённые значения
        before_data[i] = [
            song_dict[u],
            type_dict[u],
            name_dict[u],
            time_dict[u]
        ]
    # обновляем последнюю запись для пользователя
    song_dict[u] = concat.at[i, 'song_id']
    type_dict[u] = concat.at[i, 'source_type']
    name_dict[u] = concat.at[i, 'source_screen_name']
    time_dict[u] = concat.at[i, 'timestamp']

print('признаки «до» готовы')

# сбрасываем словари и готовим «последующие» значения
song_dict.clear(); type_dict.clear(); name_dict.clear(); time_dict.clear()

after_data = np.zeros((len(concat), 4))
for i in range(len(concat) - 1, -1, -1):
    u = concat.at[i, 'msno']
    if song_dict[u] is None:
        # если нет последующих, оставляем текущее и обнуляем время
        row = concat.loc[i, ['song_id', 'source_type', 'source_screen_name', 'timestamp']].values
        after_data[i] = row
        after_data[i, 3] = np.nan
    else:
        # иначе — следующие сохранённые значения
        after_data[i] = [
            song_dict[u],
            type_dict[u],
            name_dict[u],
            time_dict[u]
        ]
    # обновляем «следующие» значения для пользователя
    song_dict[u] = concat.at[i, 'song_id']
    type_dict[u] = concat.at[i, 'source_type']
    name_dict[u] = concat.at[i, 'source_screen_name']
    time_dict[u] = concat.at[i, 'timestamp']

print('признаки «после» готовы')

# добавляем новые признаки в tr и te
cols = ['song_id', 'source_type', 'source_screen_name', 'timestamp']
for idx, col in enumerate(cols):
    tr[f'before_{col}'] = before_data[:len(tr), idx]
    tr[f'after_{col}']  = after_data[:len(tr), idx]
    te[f'before_{col}'] = before_data[len(tr):, idx]
    te[f'after_{col}']  = after_data[len(tr):, idx]

# переводим категориальные before/after обратно в целые
for col in ['song_id', 'source_type', 'source_screen_name']:
    tr[f'before_{col}'] = tr[f'before_{col}'].astype(int)
    tr[f'after_{col}']  = tr[f'after_{col}'].astype(int)
    te[f'before_{col}'] = te[f'before_{col}'].astype(int)
    te[f'after_{col}']  = te[f'after_{col}'].astype(int)

# вычисляем дельты времени и логарифмируем
tr['before_timestamp'] = np.log1p(tr['timestamp'] - tr['before_timestamp'])
te['before_timestamp'] = np.log1p(te['timestamp'] - te['before_timestamp'])
tr['after_timestamp']  = np.log1p(tr['after_timestamp']  - tr['timestamp'])
te['after_timestamp']  = np.log1p(te['after_timestamp']  - te['timestamp'])

# заполняем пропуски средним
tr['before_timestamp'].fillna(tr['before_timestamp'].mean(), inplace=True)
te['before_timestamp'].fillna(te['before_timestamp'].mean(), inplace=True)
tr['after_timestamp'].fillna(tr['after_timestamp'].mean(), inplace=True)
te['after_timestamp'].fillna(te['after_timestamp'].mean(), inplace=True)

# сохраняем результаты
tr.to_csv('../temporal_data/train_id_cnt_svd_stamp_before_after.csv', index=False)
te.to_csv('../temporal_data/test_id_cnt_svd_stamp_before_after.csv', index=False)
