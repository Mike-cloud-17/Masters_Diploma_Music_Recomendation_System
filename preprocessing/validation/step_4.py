import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds

# загружаем данные
tr     = pd.read_csv('../temporal_data/train_id_cnt.csv')
te     = pd.read_csv('../temporal_data/test_id_cnt.csv')
member = pd.read_csv('../temporal_data/members_id_cnt.csv')
song   = pd.read_csv('../temporal_data/songs_id_cnt_isrc.csv')

# объединяем пары (msno, song_id) из train и test
concat = pd.concat([
    tr[['msno', 'song_id']],
    te[['msno', 'song_id']]
], ignore_index=True)

# число пользователей и песен
member_cnt = concat['msno'].max() + 1
song_cnt   = concat['song_id'].max() + 1
artist_cnt = int(song['artist_name'].max() + 1)

# ---------------- SVD для user-song ----------------
n_component = 48
print(f"Rows in concat: {len(concat)}")

# строим разреженную матрицу «прослушал / не прослушал»
data   = np.ones(len(concat))
msno   = concat['msno'].values
song_id= concat['song_id'].values
rating = sparse.coo_matrix((data, (msno, song_id)))
rating = (rating > 0).astype(float)

# вычисляем SVD
u, s, vt = svds(rating, k=n_component)
print("Singular values:", s[::-1])
s_song = np.diag(s[::-1])

# сохраняем в датафреймы
members_topics = pd.DataFrame(u[:, ::-1], columns=[f'member_component_{i}' for i in range(n_component)])
members_topics['msno'] = range(member_cnt)
member = member.merge(members_topics, on='msno', how='right')

song_topics = pd.DataFrame(vt.T[:, ::-1], columns=[f'song_component_{i}' for i in range(n_component)])
song_topics['song_id'] = range(song_cnt)
song = song.merge(song_topics, on='song_id', how='right')

# ---------------- SVD для user-artist ----------------
n_component = 16

# добавляем к concat имя артиста
concat = concat.merge(song[['song_id','artist_name']], on='song_id', how='left')
# отсекаем невалидные артисты
concat = concat[concat['artist_name'] >= 0]

msno_vals   = concat['msno'].values
artist_vals = concat['artist_name'].astype(int).values
print(f"Rows for user-artist SVD: {len(concat)}")

# строим матрицу «сколько раз слушал artist»
data_tmp   = np.ones(len(concat))
rating_tmp = sparse.coo_matrix((data_tmp, (msno_vals, artist_vals)))

# взвешиваем логарифмом и бинарным признаком
rating = np.log1p(rating_tmp) * 0.3 + (rating_tmp > 0).astype(float)

# вычисляем SVD
u_a, s_a, vt_a = svds(rating, k=n_component)
print("Artist singular values:", s_a[::-1])
s_artist = np.diag(s_a[::-1])

# сохраняем в member
members_topics = pd.DataFrame(u_a[:, ::-1], columns=[f'member_artist_component_{i}' for i in range(n_component)])
members_topics['msno'] = range(member_cnt)
member = member.merge(members_topics, on='msno', how='left')

# сохраняем в song
artist_topics = pd.DataFrame(vt_a.T[:, ::-1], columns=[f'artist_component_{i}' for i in range(n_component)])
artist_topics['artist_name'] = range(artist_cnt)
song = song.merge(artist_topics, on='artist_name', how='left')

# ---------------- dot features ----------------
# сортируем для корректного индекса
member = member.sort_values('msno')
song   = song.sort_values('song_id')

# извлекаем матрицы эмбеддингов
mem_emb   = member[[f'member_component_{i}' for i in range(48)]].values
song_emb  = song[[f'song_component_{i}'  for i in range(48)]].values
mem_a_emb = member[[f'member_artist_component_{i}' for i in range(16)]].values
song_a_emb= song[[f'artist_component_{i}' for i in range(16)]].values

# вычисляем скалярные произведения
train_dot = np.zeros((len(tr), 2))
test_dot  = np.zeros((len(te), 2))

for i in range(len(tr)):
    u_idx = tr['msno'].iat[i]
    s_idx = tr['song_id'].iat[i]
    train_dot[i,0] = np.dot(mem_emb[u_idx], s_song @ song_emb[s_idx])
    train_dot[i,1] = np.dot(mem_a_emb[u_idx], s_artist @ song_a_emb[s_idx])

for i in range(len(te)):
    u_idx = te['msno'].iat[i]
    s_idx = te['song_id'].iat[i]
    test_dot[i,0] = np.dot(mem_emb[u_idx], s_song @ song_emb[s_idx])
    test_dot[i,1] = np.dot(mem_a_emb[u_idx], s_artist @ song_a_emb[s_idx])

# сохраняем в tr и te
tr['song_embeddings_dot']   = train_dot[:,0]
tr['artist_embeddings_dot']= train_dot[:,1]
te['song_embeddings_dot']   = test_dot[:,0]
te['artist_embeddings_dot']= test_dot[:,1]

# ---------------- записываем файлы ----------------
tr.to_csv('../temporal_data/train_id_cnt_svd.csv', index=False)
te.to_csv('../temporal_data/test_id_cnt_svd.csv', index=False)
member.to_csv('../temporal_data/members_id_cnt_svd.csv', index=False)
song.to_csv('../temporal_data/songs_id_cnt_isrc_svd.csv', index=False)
