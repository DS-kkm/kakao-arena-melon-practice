import numpy as np
import pandas as pd
from xgboost import plot_importance
from xgboost import XGBClassifier
from sklearn import preprocessing

train = pd.read_json('C:/Users/chees/PycharmProjects/kakao-arena-melon-practice/arena_data/orig/train.json')
val = pd.read_json('C:/Users/chees/PycharmProjects/kakao-arena-melon-practice/arena_data/orig/val.json')

##### train: tags & songs 중첩 제거
train_svd = train[['tags','songs']]
train_svd.info()

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

##### validation: tags & songs 중첩 제거
val = val[['tags','songs']]
val.info()

# unnested songs
val_unnest = np.dstack(
    (
        np.repeat(val.tags.values, list(map(len, val.songs))),
        np.concatenate(val.songs.values)
    )
)

# unnested songs 데이터프레임 생성
val_map = pd.DataFrame(data = val_unnest[0], columns = val.columns)

# unnested tags & songs
val_unnest2 = np.dstack(
    (
        np.repeat(val_map.songs.values, list(map(len, val_map.tags))),
        np.concatenate(val_map.tags.values)
    )
)


# unnested tags & songs 데이터프레임 생성
val_map2 = pd.DataFrame(data=val_unnest2[0], columns = ['songs','tags'])
val_map2 = val_map2[['tags','songs']]
val_map2['songs'] = val_map2['songs'].astype(str)

# train data 원핫인코딩 (더미변수)
train_svd_map2_rs = train_svd_map2.sample(10000)

label_encoder = preprocessing.LabelEncoder()
onehot_encoder = preprocessing.OneHotEncoder(sparse=False)

y_train = label_encoder.fit_transform(train_svd_map2_rs['songs'])
y_train = y_train.reshape(len(y_train), 1)
y_train = onehot_encoder.fit_transform(y_train)
y_train = pd.DataFrame(y_train)
songs = train_svd_map2_rs.drop_duplicates('songs',keep='first')['songs']
y_train.columns = songs

x_train = label_encoder.fit_transform(train_svd_map2_rs['tags'])
x_train = x_train.reshape(len(x_train), 1)
x_train = onehot_encoder.fit_transform(x_train)
x_train = pd.DataFrame(x_train)
tags = train_svd_map2_rs.drop_duplicates('tags',keep='first')['tags']
x_train.columns = tags

# validation data 원핫인코딩 (더미변수)
y_val = label_encoder.fit_transform(train_svd_map2_rs['songs'])
y_val = y_train.reshape(len(y_train), 1)
y_train = onehot_encoder.fit_transform(y_train)
y_train = pd.DataFrame(y_train)
songs = train_svd_map2_rs.drop_duplicates('songs',keep='first')['songs']
y_train.columns = songs

x_train = label_encoder.fit_transform(train_svd_map2_rs['tags'])
x_train = x_train.reshape(len(x_train), 1)
x_train = onehot_encoder.fit_transform(x_train)
x_train = pd.DataFrame(x_train)
tags = train_svd_map2_rs.drop_duplicates('tags',keep='first')['tags']
x_train.columns = tags

# XGBoost
y_train = train_svd_map2_rs['songs']
x_train = train_svd_map2.drop('songs', axis=1)
y_val = val_map2['songs']
x_val = val_map2.drop('songs', axis=1)

xgb = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=4)
xgb.fit(x_train, y_train)
xgb_pred = xgb.predict(val)

# Scoring
answers = xgb.fit(x_val, y_val)
write_json(answers, "results/results.json")




