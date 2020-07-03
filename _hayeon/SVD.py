train = pd.read_json('C:/Users/chees/PycharmProjects/kakao-arena-melon-practice/res/train.json', typ = 'frame')
train_svd = train[['tags','songs']]
train_svd.info()

# ##### songs 벡터화, 중복값 제거
# # train_svd에서 songs 추출
# songs = train_svd[['songs']]
#
# # songs의 여러 리스트 하나로 합치기
# songs = np.concatenate((songs,), axis=None)
# songs = list(it.chain(*songs))   # itertools를 사용하여 중첩리스트를 하나의 리스트로 반환
#                                         # length = 5,285,871
# songs_uq = set(songs)       # set을 사용하여 list를 집합으로 반환 (단, 순서가 뒤죽박죽이므로 순서가 중요하면 for문 사용)
# songs_uq = list(songs_uq)   # songs의 중복값 제거
#                             # length = 615,142 (4,670,729개의 중복값 제거됨)
#
# ##### tags 벡터화, 중복값 제거
# # train_svd에서 tags 추출
# tags = train_svd[['tags']]
#
# # tags의 여러 리스트 하나로 합치기
# tags = np.concatenate((tags,), axis=None)
# tags = list(itertools.chain(*tags))     # length = 476,331
#
# tags_uq = set(tags)
# tags_uq = list(tags_uq)     # length = 29,160

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

# ##### 행: tags & 열: songs 빈도 테이블 생성 (for SVD)
# table = pd.crosstab(train_svd_map2.songs, train_svd_map2.tags)  # 이건 data가 너무 커서 오류남......
#
# tags, songs = pd.factorize(list(zip(*map(train_svd_map2.get, train_svd_map2))))
# result = dict(zip(songs, np.bincount(tags)))
# table = pd.Series(result).unstack(fill_value=0)     # 마찬가지로 오류...
#
# train_svd_map2.to_excel('train_svd_map2.xlsx')


##### train_svd_map2 데이터 일부 추출
train_svd_map2_str = train_svd_map2.loc[0:9999]
table = pd.crosstab(train_svd_map2_str.songs, train_svd_map2_str.tags)

##### SVD
U, S, VT = np.linalg.svd(table,full_matrices=False)

S = np.diag(S) # diagonal matrix로 변환

table2 = np.dot(np.dot(U, S), VT)
table2 = pd.DataFrame(table2)
table2.columns = table.columns
table2.index = table.index

##### Modeling
def recommendation():
return
