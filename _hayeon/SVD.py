import json
import numpy as np
import pandas as pd
import itertools as it

train = pd.read_json('C:/Users/chees/PycharmProjects/kakao-arena-melon-practice/arena_data/orig/train.json', typ = 'frame')
val = pd.read_json('C:/Users/chees/PycharmProjects/kakao-arena-melon-practice/arena_data/orig/val.json', typ = 'frame')

train_svd = train[['tags','songs']]
train_svd.info()

##### tags & songs 중첩 제거
# unnested songs
train_svd_unnest = np.dstack(
    (
        np.repeat(train_svd.tags.values, list(map(len, train_svd.songs))),
        np.concatenate(train_svd.songs.values)
    )
)

# unnested songs 데이터프레임 생성
train_svd_map = pd.DataFrame(data = train_svd_unnest[0], columns = train_svd.columns)

# unnested tags & songs
train_svd_unnest2 = np.dstack(
    (
        np.repeat(train_svd_map.songs.values, list(map(len, train_svd_map.tags))),
        np.concatenate(train_svd_map.tags.values)
    )
)

# unnested tags & songs 데이터프레임 생성
train_svd_map2 = pd.DataFrame(data=train_svd_unnest2[0], columns = ['songs','tags'])
train_svd_map2 = train_svd_map2[['tags','songs']]
train_svd_map2['songs'] = train_svd_map2['songs'].astype(str)


##### train_svd_map2 데이터 일부 추출
train_svd_map2_str = train_svd_map2.loc[0:9999]
table = pd.crosstab(train_svd_map2_str.songs, train_svd_map2_str.tags)

##### SVD
U, S, VT = np.linalg.svd(table,full_matrices=False)

S_diag = np.diag(S) # diagonal matrix로 변환
S_prop = S / sum(S)

# Singular value 90% quantile값 찾기
# 1. == 사용 (안됨)
np.where(S_prop == np.percentile(S_prop, 90))

# 2. isclose 함수 사용 (안됨)
import math to math
S_prop_qt = np.arange(len(S_prop))
for i in range(0, len(S_prop)):
    S_prop_qt[i] = math.isclose(S_prop[i] , np.percentile(S_prop, 90))

table2 = np.dot(np.dot(U, S_diag), VT)
table2 = pd.DataFrame(table2)

# table2.columns = table.columns
# table2.index = table.index

# test data의 tags & songs 공백인 행 추출하여 각각 데이터프레임 생성
val_tags_pred = val.loc[[0]]
for i in range(0, len(val)):
if len(val.tags[i])==0:
    val_tags_pred = pd.concat([val_tags_pred,val.loc[[i]]])

val_songs_pred = val.loc[[5]]
for i in range(0, len(val)):
if len(val.songs[i])==0:
    val_songs_pred = pd.concat([val_songs_pred,val.loc[[i]]])

##### Modeling
def recommendation(train,train_pred):
    songs_index = pd.DataFrame(index=range(0, len(train)), columns=[])
    # 태그를 기반으로 songs 예측하기
    for i in range(0, len(train_pred)):
        for j in  range(0, len(train)):
            train_pred.index[i] == train.tags[j]



# 일치하는 태그를 찾아서
# 그 태그의 songs의 목록을 추출하고
# 그 중에 없는 songs을 추출

return
