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
#이런 행렬을 만들 계획

#song의 개수만 따로 새로운 song_len으로 데이터프레임에 추가
alist = [len(i) for i in train_data["songs"]]
train_data["song_len"] = alist
#print(train_data.head())

#train데이터에서 id와 tags만 추출해 데이터프레임 형성
id_tags = pd.DataFrame(
    train_data['tags'].values.tolist(),
    index = train_data['id']
).stack().reset_index().drop('level_1',axis=1).rename(columns={0:'tag'})
#train데이터에서 id와 song_len만 추출
id_like = train_data[['id','song_len']]
#두 개 id를 기준으로 합침
playlist_1st = pd.merge(id_tags, id_like, on='id')
#print(playlist_1st.head(50))

#이대로 돌리면 데이터메모리가 너무 커서 memory error가 뜨므로 데이터 크기를 줄여준다.
#함수 check_dtypes() 이용
df = pd.get_dummies(playlist_1st,columns=['tag'])
data_types = check_dtypes(df)
df = df.astype(data_types)
print(df.head(50))