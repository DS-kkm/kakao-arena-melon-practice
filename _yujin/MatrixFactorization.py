import json
import os
import re
import numpy as np
import pandas as pd
import scipy

#song_meta data와 train data를 이용해 matrix를 만들어 matrix factorization을 할 것입니다.
#장르데이터는 song_meta.json에 포함되어 있으므로 별도로 이용하지는 않겠습니다.
song_meta = pd.read_json('../res/song_meta.json',typ='frame')
train = pd.read_json('../res/train.json',typ='frame')

#좋아요횟수 이용
#피벗테이블 형성 >> 에러
playlist_id_song = train.pivot(
    index='id',
    columns='songs',
    values='like_cnt'
)
print(playlist_id_song)

matrix = playlist_id_song.as_matrix() #피벗테이블을 행렬로 변환
id_score_mean = np.mean(matrix,axis=1) #사용자의 좋아요횟수 평균
matrix_id_mean = matrix - id_score_mean.reshape(-1,1) #(사용자-노래 평점 행렬) - (사용자의 좋아요횟수 평균)
print(matrix)
pd.DataFrame(matrix_id_mean,columns=playlist_id_song.columns).head()

#Python scipy에서 제공해주는 svd 이용. A=U*SIGMA*Vt
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

svd_playlist = pd.DataFrame(svd_matrix_playlist,columns=playlist_id_song.columns)
print(svd_playlist.head())

#함수 생성
#인자:name_id,song,like_num
#svd로 나온 결과에서 사용자id당 좋아요횟수가 많은 데이터 순으로 정렬
#사용자가 들은 song은 제외. 사용자가 안 들은 song 중 좋아요횟수가 높은 것을 추천
def recommend_song(svd_playlist,name_id,song,like_num,num_recommendations=10):
    id_row_num = name_id - 1 #index라서 0부터 시작하니까
    sorted_playlist = svd_playlist.iloc[id_row_num].sort_values(ascending=False) #좋아요횟수 많은 순으로 정렬
    playlist_train = train[['id','songs']] #train데이터에서 id와 songs추출
    id_same = playlist_train[id==name_id] #train데이터의 id랑 name_id가 같은 것 따로 데이터명 부여
    id_past = id_same.merge(song_meta, on='id') #song_meta데이터랑 합친다
    recommendations = song_meta[~song_meta['id'].isin(id_past['songs'])]
    #song_meta데이터에서 사용자가 이미 들은 노래를 제외한 노래데이터 추출
    recommendations = recommendations.merge(pd.DataFrame(sorted_playlist).reset_index(),on='songs')
    #좋아요횟수 높은 순으로 정렬된 데이터와 합친다 (왜 합치지)
    recommendations = recommendations.rename(columns = {id_row_num:'Predictions'}).sort_values('Predictions',ascending=False)
    return id_past, recommendations

