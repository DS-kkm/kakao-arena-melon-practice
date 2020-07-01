import json
import os
import re
import numpy as np
import pandas as pd

#내가 하고 싶은 것
#1) train데이터에서 id,tags,like_cnt만 추출
# (사실 like_cnt 대신 songs를 추출해서 하고 싶었으나 데이터프레임 속 songs 개수 세는법 몰라서 포기)
#2) id가 선호하는(like_cnt가 높은) tags의 다른 곡들 중 듣지 않았던 곡을 추천
song_meta = pd.read_json('../res/song_meta.json',encoding='utf-8')
train_data = pd.read_json('../res/train.json',encoding='utf-8')
train_data = train_data[['id','tags','like_cnt']]
print(train_data.head())

#id와 tag 간 데이터프레임 생성
id_tags = pd.DataFrame(
    train_data['tags'].values.tolist(),
    index = train_data['id']
).stack().reset_index().drop('level_1',axis=1).rename(columns={0:'tag'})
#train데이터에서 id와 좋아요횟수만 추출
id_like = train_data[['id','like_cnt']]
#두 개 id를 기준으로 합침
playlist_1st = pd.merge(id_tags, id_like, on='id')
print(playlist_1st.head(50))

#피벗테이블을 만들려 하였으나 데이터프레임 크기 개크다고 까임
playlist_pivot = pd.get_dummies(playlist_1st,columns=['tag'])
