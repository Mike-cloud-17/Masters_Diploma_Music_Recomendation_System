import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds

# загружаем предобработанные данные
tr     = pd.read_csv('../temporal_data/train_id_cnt.csv')
te     = pd.read_csv('../temporal_data/test_id_cnt.csv')
member = pd.read_csv('../temporal_data/members_id_cnt.csv')
song   = pd.read_csv('../temporal_data/songs_id_cnt_isrc.csv')

# объединяем все прослушивания (user-song) из train и test
concat = pd.concat([
    tr[['msno', 'song_id']],
    te[['msno', 'song_id']]
], ignore_index=True)

# определяем размеры матриц
member_cnt = concat['msno'].max() + 1
song_cnt   = concat['song_id'].max() + 1
artist_cnt = int(song['artist_name'].max() + 1)

# === SVD для матрицы user-song ===
n_component = 48
print(f"Всего записей в concat: {len(concat)}")

# формируем разреженную матрицу взаимодействий
data    = np.ones(len(concat))
msno    = concat['msno'].values
song_id = concat['song_id'].values
rating  = sparse.coo_matrix((data, (msno, song_id)))
rating  = (rating > 0).astype(float)

# вычисляем первые n_component сингулярных значений и векторов
u, s, vt = svds(rating, k=n_component)
s        = s[::-1]   # упорядочим по убыванию
print("Сингулярные значения (user-song):", s)

# создаём диагональную матрицу для дальнейших dot-умножений
s_song = np.diag(s)

# составляем темы пользователей
members_topics = pd.DataFrame(
    u[:, ::-1],
    columns=[f'member_component_{i}' for i in range(n_component)]
)
members_topics['msno'] = np.arange(member_cnt)
member = member.merge(members_topics, on='msno', how='right')

# составляем темы песен
song_topics = pd.DataFrame(
    vt.T[:, ::-1],
    columns=[f'song_component_{i}' for i in range(n_component)]
)
song_topics['song_id'] = np.arange(song_cnt)
song = song.merge(song_topics, on='song_id', how='right')

# === SVD для матрицы user-artist ===
n_component = 16

# добавляем колонку artist_name в interactions
concat = concat.merge(song[['song_id','artist_name']], on='song_id', how='left')
concat = concat[concat['artist_name'] >= 0]

msno_vals   = concat['msno'].values
artist_vals = concat['artist_name'].astype(int).values
data        = np.ones(len(concat))
rating_tmp  = sparse.coo_matrix((data, (msno_vals, artist_vals)),
                                shape=(member_cnt, artist_cnt))

# комбинируем логарифм и бинарный признак
rating = np.log1p(rating_tmp) * 0.3 + (rating_tmp > 0).astype(float)

u2, s2, vt2 = svds(rating, k=n_component)
s2          = s2[::-1]
print("Сингулярные значения (user-artist):", s2)
s_artist    = np.diag(s2)

# темы пользователей по артистам
members_topics2 = pd.DataFrame(
    u2[:, ::-1],
    columns=[f'member_artist_component_{i}' for i in range(n_component)]
)
members_topics2['msno'] = np.arange(member_cnt)
member = member.merge(members_topics2, on='msno', how='left')

# темы артистов
artist_topics = pd.DataFrame(
    vt2.T[:, ::-1],
    columns=[f'artist_component_{i}' for i in range(n_component)]
)
artist_topics['artist_name'] = np.arange(artist_cnt)
song = song.merge(artist_topics, on='artist_name', how='left')

# === вычисляем dot-признаки для каждого взаимодействия ===
# готовим эмбеддинги
member = member.sort_values('msno').reset_index(drop=True)
song   = song.sort_values('song_id').reset_index(drop=True)

user_emb       = member[[f'member_component_{i}' for i in range(48)]].values
item_emb       = song  [[f'song_component_{i}'   for i in range(48)]].values
user_art_emb   = member[[f'member_artist_component_{i}' for i in range(16)]].values
item_art_emb   = song  [[f'artist_component_{i}'       for i in range(16)]].values

train_dot = np.zeros((len(tr), 2))
test_dot  = np.zeros((len(te), 2))

# рассчитываем скалярные произведения с учётом сингулярных значений
for idx, row in tr.iterrows():
    u_idx = int(row['msno'])
    s_idx = int(row['song_id'])
    train_dot[idx, 0] = user_emb[u_idx]     @ (s_song   @ item_emb[s_idx])
    train_dot[idx, 1] = user_art_emb[u_idx] @ (s_artist @ item_art_emb[s_idx])

for idx, row in te.iterrows():
    u_idx = int(row['msno'])
    s_idx = int(row['song_id'])
    test_dot[idx, 0] = user_emb[u_idx]     @ (s_song   @ item_emb[s_idx])
    test_dot[idx, 1] = user_art_emb[u_idx] @ (s_artist @ item_art_emb[s_idx])

tr['song_embeddings_dot']   = train_dot[:, 0]
tr['artist_embeddings_dot'] = train_dot[:, 1]
te['song_embeddings_dot']   = test_dot[:, 0]
te['artist_embeddings_dot'] = test_dot[:, 1]

# сохраняем результаты
tr.to_csv('../temporal_data/train_id_cnt_svd.csv', index=False)
te.to_csv('../temporal_data/test_id_cnt_svd.csv',  index=False)
member.to_csv('../temporal_data/members_id_cnt_svd.csv', index=False)
song.to_csv('../temporal_data/songs_id_cnt_isrc_svd.csv', index=False)
