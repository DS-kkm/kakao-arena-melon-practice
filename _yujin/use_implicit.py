import json
import os
import re
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import uniform
from tqdm import tqdm

song_meta = pd.read_json('../res/song_meta.json',encoding='utf-8') #곡 정보가 들어있는 데이터. song id, artist id, genre 등의 정보 있다.
Train_data = pd.read_json('../res/train.json',encoding='utf-8') #예비테스트용 데이터. playlist id, 그 playlist 당 song id리스트들&태그& 등등이 있다. (장르x. song은 리스트로 표현되어있음)
train_data = Train_data.sample(frac=0.30)
train_data2 = Train_data.drop(train_data.index)

song_list = song_meta[['id']]
id_list = train_data[['id']]

#song이 그 플레이리스트에 존재하면 1, 존재하지 않으면 0 부여하는 데이터리스트 생성
song_meta_ = song_meta['id'].tolist()
songs = [i for i in train_data['songs']]
id_train_ = [i for i in train_data['id']]

dic = {}
for idx, row in tqdm(train_data.iterrows(), total=len(train_data)):
    dic[idx] = row.songs

data_list = []
for i in tqdm(song_meta_):
    for j in id_train_:
        if song_meta_[i] in dic[j]:
            data_list.append(1)
        else: data_list.append(0)
print(data_list)
'''
row_ind = np.array(id_list)
col_ind = np.array(song_list)
data = np.array(dic)
matrix_playlist = sparse.coo_matrix((dic,(row_ind,col_ind)))
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
'''