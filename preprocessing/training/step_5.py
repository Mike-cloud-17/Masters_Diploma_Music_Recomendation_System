import numpy as np
import pandas as pd
from collections import defaultdict

# Загружаем результаты предыдущего шага
tr   = pd.read_csv('../temporal_data/train_id_cnt_svd.csv')
te   = pd.read_csv('../temporal_data/test_id_cnt_svd.csv')
mem  = pd.read_csv('../temporal_data/members_id_cnt_svd.csv')
song = pd.read_csv('../temporal_data/songs_id_cnt_isrc_svd.csv')

# Объединяем train и test для общего расчёта по порядку событий
concat = pd.concat([
    tr[['msno', 'song_id']],
    te[['msno', 'song_id']]
], ignore_index=True)
# Нумеруем события в порядке следования
concat['timestamp'] = np.arange(len(concat))

# Считаем количество предыдущих и будущих прослушиваний в различных окнах
window_sizes = [10, 25, 500, 5000, 10000, 50000]
msno_list = concat['msno'].values
song_list = concat['song_id'].values

def get_window_cnt(values, idx, window):
    left  = max(0, idx - window)
    right = min(len(values), idx + window)
    before = (values[left:idx] == values[idx]).sum()
    after  = (values[idx:right] == values[idx]).sum()
    return before, after

for w in window_sizes:
    before_msno = np.zeros(len(concat), dtype=int)
    before_song = np.zeros(len(concat), dtype=int)
    after_msno  = np.zeros(len(concat), dtype=int)
    after_song  = np.zeros(len(concat), dtype=int)
    for i in range(len(concat)):
        b_ms, a_ms = get_window_cnt(msno_list,  i, w)
        b_so, a_so = get_window_cnt(song_list, i, w)
        before_msno[i], after_msno[i] = b_ms, a_ms
        before_song[i], after_song[i] = b_so, a_so
    concat[f'msno_{w}_before_cnt'] = before_msno
    concat[f'song_{w}_before_cnt'] = before_song
    concat[f'msno_{w}_after_cnt']  = after_msno
    concat[f'song_{w}_after_cnt']  = after_song
    print(f'Окно {w} обработано')

# Считаем накопленные до текущего момента повторения
msno_hist = defaultdict(int)
song_hist = defaultdict(int)
till_msno = np.zeros(len(concat), dtype=int)
till_song = np.zeros(len(concat), dtype=int)

for i in range(len(concat)):
    u = msno_list[i]; s = song_list[i]
    till_msno[i] = msno_hist[u]
    till_song[i] = song_hist[s]
    msno_hist[u] += 1
    song_hist[s] += 1

concat['msno_till_now_cnt'] = till_msno
concat['song_till_now_cnt'] = till_song
print('Накопленные счётчики готовы')

# Переводим порядковый индекс в unix-время
def timestamp_map(x):
    if x < 7377418:
        return (x - 0) / (7377417 - 0) * (1484236800 - 1471190400) + 1471190400
    else:
        return (x - 7377417) / (9934207 - 7377417) * (1488211200 - 1484236800) + 1484236800

concat['timestamp'] = concat['timestamp'].apply(timestamp_map)

# Вычисляем для каждого пользователя и песни среднее и стандартное отклонение времени
msno_stats = concat.groupby('msno')['timestamp']
mem['msno_timestamp_mean'] = mem['msno'].map(msno_stats.mean())
mem['msno_timestamp_std']  = mem['msno'].map(msno_stats.std())

song_stats = concat.groupby('song_id')['timestamp']
song['song_timestamp_mean'] = song['song_id'].map(song_stats.mean())
song['song_timestamp_std']  = song['song_id'].map(song_stats.std())

print('Временные статистики готовы')

# Логарифмируем все рассчитанные счётчики
features = ['msno_till_now_cnt', 'song_till_now_cnt']
for w in window_sizes:
    features += [
        f'msno_{w}_before_cnt', f'song_{w}_before_cnt',
        f'msno_{w}_after_cnt',  f'song_{w}_after_cnt'
    ]
for feat in features:
    concat[feat] = np.log1p(concat[feat])

# Разбиваем обратно на train и тест и сохраняем
all_feats = ['timestamp'] + features
arr = concat[all_feats].values

for idx, feat in enumerate(all_feats):
    tr[feat] = arr[:len(tr), idx]
    te[feat] = arr[len(tr):, idx]

tr.to_csv('../temporal_data/train_id_cnt_svd_stamp.csv', index=False)
te.to_csv('../temporal_data/test_id_cnt_svd_stamp.csv', index=False)
mem.to_csv('../temporal_data/members_id_cnt_svd_stamp.csv', index=False)
song.to_csv('../temporal_data/songs_id_cnt_isrc_svd_stamp.csv', index=False)
