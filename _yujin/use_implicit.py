# import json
# import os
# import re
import numpy as np
import pandas as pd
# from scipy import sparse
# from scipy.stats import uniform
from tqdm import tqdm
#
song_meta = pd.read_json('../res/song_meta.json',encoding='utf-8')
Train_data = pd.read_json('../res/train.json',encoding='utf-8')
train_data = Train_data.sample(frac=0.50)
val_data = pd.read_json('../res/val.json',encoding='utf-8')
#
#
# #song이 그 플레이리스트에 존재하면 1, 존재하지 않으면 0 부여하는 데이터리스트 생성
id_songs = (
    train_data[['id', 'songs']]
    .explode('songs')
    .assign(value=1)
    .rename(columns={'id':'user_id', 'songs':'item_id'})
)
#
# # range of int32
# assert id_songs['user_id'].max() < 2147483647
# assert id_songs['item_id'].max() < 2147483647
#
# id_songs['user_id'] = id_songs['user_id'].astype(np.int32)
# id_songs['item_id'] = id_songs['item_id'].astype(np.int32)
# id_songs['value'] = id_songs['value'].astype(np.int8)
#
# #행렬이 너무 커서 빈도수 적은 요인들 제거
# id_songs.user_id.nunique() * id_songs.item_id.nunique()
#
# while True:
#     prev = len(id_songs)
#
#     # 5곡 이상 가진 플레이 리스트만
#     user_count = id_songs.user_id.value_counts()
#     id_songs = id_songs[id_songs.user_id.isin(user_count[user_count >= 5].index)]
#
#     # 5번 이상 등장한 곡들만
#     item_count = id_songs.item_id.value_counts()
#     id_songs = id_songs[id_songs.item_id.isin(item_count[item_count >= 5].index)]
#
#     cur = len(id_songs)
#
#     if prev == cur: break
#
# id_songs.user_id.nunique() * id_songs.item_id.nunique() #10%정도로 감소함
#
# from scipy.sparse import csr_matrix
# from pandas.api.types import CategoricalDtype
#
# person_c = CategoricalDtype(sorted(id_songs.user_id.unique()), ordered=True)
# thing_c = CategoricalDtype(sorted(id_songs.item_id.unique()), ordered=True)
#
# row = id_songs.user_id.astype(person_c).cat.codes
# col = id_songs.item_id.astype(thing_c).cat.codes
# sparse_matrix = csr_matrix((id_songs["value"], (row, col)), \
#                            shape=(person_c.categories.size, thing_c.categories.size))
#
#
# import implicit
#
# # initialize a model
# model = implicit.als.AlternatingLeastSquares(factors=100)
#
# # train the model on a sparse matrix of item/user/confidence weights
# model.fit(sparse_matrix.T)
#
# # recommend items for a user
# user_items = sparse_matrix.tocsr()
#
# # 인기곡
# top_songs = id_songs.item_id.value_counts().nlargest(100)
# item_index = np.where(thing_c.categories==top_songs.index[0])
#
# # find related items
# related = model.similar_items(item_index[0][0])

from numpy import dot
from numpy.linalg import norm

song_list = song_meta['id']
# print(song_list)


val_vec = np.zeros((len(song_list)))
# print(val_vec)

dic = {}
for i, v in enumerate(song_list):
    dic[v] = i



for slist in tqdm(val_data['songs']):
    for idx in slist:
        if idx in dic:
            val_vec[dic[idx]] = 1

# print(val_vec)

def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))

train_val = []
a = cos_sim(id_songs.user_id,val_vec)
train_val = train_val.append(a)

print(train_val)
