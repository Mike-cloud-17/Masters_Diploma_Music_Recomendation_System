import numpy as np
import pandas as pd
from collections import defaultdict

# загружаем данные после SVD
tr   = pd.read_csv('../temporal_data/train_id_cnt_svd.csv')
te   = pd.read_csv('../temporal_data/test_id_cnt_svd.csv')
mem  = pd.read_csv('../temporal_data/members_id_cnt_svd.csv')
song = pd.read_csv('../temporal_data/songs_id_cnt_isrc_svd.csv')

# создаём единый датафрейм с непрерывным индексом и назначаем «timestamp» как порядковый номер
concat = pd.concat(
    [tr[['msno','song_id']], te[['msno','song_id']]],
    ignore_index=True
)
concat['timestamp'] = np.arange(len(concat))

# === оконные счётчики до и после текущей позиции ===
window_sizes = [10, 25, 500, 5000, 10000, 50000]
msno_list  = concat['msno'].values
song_list  = concat['song_id'].values

def get_window_cnt(values, idx, window_size):
    lower = max(0, idx - window_size)
    upper = min(len(values), idx + window_size)
    before = (values[lower:idx] == values[idx]).sum()
    after  = (values[idx:upper] == values[idx]).sum()
    return before, after

for w in window_sizes:
    # инициализируем массивы для каждого окна
    msno_before = np.zeros(len(concat), dtype=int)
    song_before = np.zeros(len(concat), dtype=int)
    msno_after  = np.zeros(len(concat), dtype=int)
    song_after  = np.zeros(len(concat), dtype=int)
    # проходим по всем строкам
    for i in range(len(concat)):
        mb, ma = get_window_cnt(msno_list,  i, w)
        sb, sa = get_window_cnt(song_list,  i, w)
        msno_before[i], msno_after[i] = mb, ma
        song_before[i], song_after[i] = sb, sa
    # сохраняем новые признаки
    concat[f'msno_{w}_before_cnt'] = msno_before
    concat[f'song_{w}_before_cnt'] = song_before
    concat[f'msno_{w}_after_cnt']  = msno_after
    concat[f'song_{w}_after_cnt']  = song_after
    print(f'Window size {w} done.')

# === накопительный счётчик до текущей позиции ===
msno_acc = defaultdict(int)
song_acc = defaultdict(int)
msno_till = np.zeros(len(concat), dtype=int)
song_till = np.zeros(len(concat), dtype=int)

for i, (u, s) in enumerate(zip(msno_list, song_list)):
    msno_till[i] = msno_acc[u]
    song_till[i] = song_acc[s]
    msno_acc[u] += 1
    song_acc[s] += 1

concat['msno_till_now_cnt'] = msno_till
concat['song_till_now_cnt'] = song_till
print('Till-now count done.')

# === преобразование «timestamp» из порядкового номера в unix-время ===
def timestamp_map(x):
    # если индекс до границы, маппим на первый интервал
    if x < 7377418:
        return (x - 0.0)/(7377417.0 - 0.0)*(1484236800.0 - 1471190400.0) + 1471190400.0
    # иначе на второй
    return (x - 7377417.0)/(9934207.0 - 7377417.0)*(1488211200.0 - 1484236800.0) + 1484236800.0

concat['timestamp'] = concat['timestamp'].apply(timestamp_map)

# === статистики по timestamp для пользователей и песен ===
msno_mean = concat.groupby('msno')['timestamp'].mean().to_dict()
msno_std  = concat.groupby('msno')['timestamp'].std().to_dict()
song_mean = concat.groupby('song_id')['timestamp'].mean().to_dict()
song_std  = concat.groupby('song_id')['timestamp'].std().to_dict()

mem['msno_timestamp_mean'] = mem['msno'].map(msno_mean)
mem['msno_timestamp_std']  = mem['msno'].map(msno_std)
song['song_timestamp_mean'] = song['song_id'].map(song_mean)
song['song_timestamp_std']  = song['song_id'].map(song_std)

print('Variance done.')

# === логарифмируем счётчики и сохраняем новые файлы ===
features = ['msno_till_now_cnt','song_till_now_cnt']
for w in window_sizes:
    features += [f'msno_{w}_before_cnt', f'song_{w}_before_cnt',
                 f'msno_{w}_after_cnt',  f'song_{w}_after_cnt']
# логарифм
for feat in features:
    concat[feat] = np.log1p(concat[feat])

# записываем обратно в train/test
for i, feat in enumerate(['timestamp'] + features):
    tr[feat] = concat.iloc[:len(tr),   i].values
    te[feat] = concat.iloc[len(tr):,    i].values

# сохраняем
tr.to_csv('../temporal_data/train_id_cnt_svd_stamp.csv', index=False)
te.to_csv('../temporal_data/test_id_cnt_svd_stamp.csv',  index=False)
mem.to_csv('../temporal_data/members_id_cnt_svd_stamp.csv', index=False)
song.to_csv('../temporal_data/songs_id_cnt_isrc_svd_stamp.csv', index=False)
