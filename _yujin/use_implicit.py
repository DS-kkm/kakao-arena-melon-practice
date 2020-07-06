import json
import os
import re
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import uniform

song_meta = pd.read_json('../res/song_meta.json',encoding='utf-8')
train_data = pd.read_json('../res/train.json',encoding='utf-8')

song_list = song_meta[['id']]
id_list = train_data[['id']]

#song이 그 플레이리스트에 존재하면 1, 존재하지 않으면 0 부여하는 데이터리스트 생성
song_meta_ = [i for i in song_meta['id']]
song_meta_list = []
for i in song_meta_:
    song_meta_list.append(i)

songs = [i for i in train_data['songs']]
songs_list = []
for i in songs:
    songs_list.append(i)
id_train_ = [i for i in train_data['id']]
id_train_list = []
for i in id_train_:
    id_train_list.append(i)
dict = {}
for i in id_train_list:
    for j in songs_list:
        dict[i] = j

data_list = []
for i in song_meta_list:
    for j in id_train_list:
        if song_meta_list[i] in dict[j]:
            data_list.append(1)
        else: data_list.append(0)


row_ind = np.array(id_list)
col_ind = np.array(song_list)
data = np.array(data_list)
matrix_playlist = sparse.coo_matrix((data,(row_ind,col_ind)))
print(matrix_playlist)

#플레이리스트id와 song id만 따로 추출
userid = train_data['id']
songid = song_meta['id']

#implicit 패키지 이용
import implicit

#initialize a model
playlist = implicit.als.AlternatingLeastSquares(factors=50)
#train the model on a sparse matrix of data
playlist.fit(matrix_playlist)
#recommend items for a user
user_playlist = matrix_playlist.T.tocsr()
recommendations = playlist.recommend(userid, user_playlist)
#find related items
related = playlist.similar_items(songid)
