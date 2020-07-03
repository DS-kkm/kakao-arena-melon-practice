import json
import os
import re
import numpy as np
import pandas as pd

#내가 하고 싶은 것
#1) train데이터에서 id,tags,songs만 추출
#songs는 song의 개수로 치환
#2) id가 선호하는(song개수가 높은) tags의 다른 곡들 중 듣지 않았던 곡을 추천
song_meta = pd.read_json('../res/song_meta.json',encoding='utf-8')
train_data = pd.read_json('../res/train.json',encoding='utf-8')

#song_meta에서 써야하는 데이터는 없나?
#train_data에서 태그를 분해? >> column이 너무 많아지는 문제는 어떻게...
#train_data[[id,tags,song개수]]
#      tag1 tag2 ...
# id1  곡수 곡수 ...
#이런 행렬을 만들 계획인데 ...

#song의 개수만 따로 새로운 song_len으로 데이터프레임에 추가
alist = [len(i) for i in train_data["songs"]]
train_data["song_len"] = alist
#print(train_data.head())

#train데이터에서 id,tags,song_len만 추출
item_user_data = train_data[['id','tags','song_len']]
#id만 따로 추출
userid = train_data['id']
songid = song_meta['id']

#print(songid)
#implicit 패키지 이용
import implicit

#initialize a model
playlist = implicit.als.AlternatingLeastSquares(factors=50)
#train the model on a sparse matrix of data
playlist.fit(item_user_data)
#recommend items for a user
user_playlist = item_user_data.T.tocsr()
recommendations = playlist.recommend(userid, user_playlist)
#find related items
related = playlist.similar_items(songid)
