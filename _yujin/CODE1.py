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
from _yujin import function_checkdtypes
df = pd.get_dummies(playlist_1st,columns=['tag'])
data_types = function_checkdtypes.check_dtypes(df)
df = df.astype(data_types)
print(df.head(50))

id_score_mean = np.mean(df,axis=1) #사용자의 song개수 평균
matrix_id_mean = df - id_score_mean.reshape(-1,1) 
pd.DataFrame(matrix_id_mean,columns=df.columns).head()

#Python scipy에서 제공해주는 svd 이용. A=U*SIGMA*Vt
import scipy
U, sigma, Vt = svds(matrix_id_mean,k=12)
print(U.shape)
print(sigma.shape)
print(Vt.shape)
#이 때 sigma행렬은 0이 아닌 값만 1차원 행렬로 표현된 상태
#->> 0이 포함된 대칭행렬로 변환 시 numpy diag이용
sigma = np.dia(sigma)
print(sigma.shape)
print(sigma[0])

#원 행렬로 복구 >> np.dot(np.dot(U,sigma),Vt) 이용
svd_matrix_playlist = np.dot(np.dot(U, sigma), Vt) + id_score_mean.reshape(-1,1)
                                                    #svd하기 전 빼주었으니 다시 더해준다.

svd_playlist = pd.DataFrame(svd_matrix_playlist,columns=df.columns)
print(svd_playlist.head())

#함수 생성
#인자:name_id,tag,song
#svd로 나온 결과에서 사용자id당 song개수가 많은 데이터 순으로 정렬
#사용자가 들은 song은 제외. 사용자가 안 들은 song 중 좋아요횟수가 높은 것을 추천
def recommend_song(svd_playlist,name_id,song,tag,num_recommendations=10):
    id_row_num = name_id - 1 #index라서 0부터 시작하니까
    sorted_playlist = svd_playlist.iloc[id_row_num].sort_values(ascending=False) #song개수 많은 순으로 정렬
    playlist_train = train_data[['id','tags']] #train데이터에서 id와 tags추출
    id_same = playlist_train[id==name_id] #train데이터의 id랑 name_id가 같은 것 따로 데이터명 부여
    id_past = id_same.merge(song_meta, on='id') #song_meta데이터랑 합친다
    recommendations = song_meta[~song_meta['id'].isin(id_past['songs'])]
    #song_meta데이터에서 사용자가 이미 들은 노래를 제외한 노래데이터 추출
    recommendations = recommendations.merge(pd.DataFrame(sorted_playlist).reset_index(),on='songs')
    #song개수 높은 순으로 정렬된 데이터와 합친다 (왜 합치지)
    recommendations = recommendations.rename(columns = {id_row_num:'Predictions'}).sort_values('Predictions',ascending=False)
    return id_past, recommendations

