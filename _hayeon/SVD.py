train_svd = train[['tags','songs']]
train_svd.info()

# train_svd에서 songs 추출
songs = train_svd[['songs']]

# songs의 여러 리스트 하나로 합치기
songs = np.concatenate((songs,), axis=None)
songs = list(itertools.chain(*songs))   # itertools를 사용하여 중첩리스트를 하나의 리스트로 반환
                                        # length = 5,285,871

songs_uq = set(songs)       # set을 사용하여 list를 집합으로 반환 (단, 순서가 뒤죽박죽이므로 순서가 중요하면 for문 사용)
songs_uq = list(songs_uq)   # songs의 중복값 제거
                            # length = 615,142 (4,670,729개의 중복값 제거됨)

# train_svd에서 tags 추출
tags = train_svd[['tags']]

# tags의 여러 리스트 하나로 합치기
tags = np.concatenate((tags,), axis=None)
tags = list(itertools.chain(*tags))     # length = 476,331

tags_uq = set(tags)
tags_uq = list(tags_uq)     # length = 29,160

# unnested songs
train_svd_unnest = np.dstack(
    (
        np.repeat(train_svd.tags.values, list(map(len, train_svd.songs))),
        np.concatenate(train_svd.songs.values)
    )
)

# unnested songs 데이터프레임 생성
train_svd_map = pd.DataFrame(data = train_svd_unnest[0], columns = train_svd.columns)

# unnested tags
train_svd_unnest2 = np.dstack(
    (
        np.repeat(train_svd_map.songs.values, list(map(len, train_svd_map.tags))),
        np.concatenate(train_svd_map.tags.values)
    )
)

# unnested tags 데이터프레임 생성
train_svd_map2 = pd.DataFrame(data=train_svd_unnest2[0], columns = ['songs','tags'])
train_svd_map2 = train_svd_map2[['tags','songs']]
