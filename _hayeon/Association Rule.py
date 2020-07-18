import json
import numpy as np
import pandas as pd
import itertools

train = pd.read_json('C:/Users/chees/PycharmProjects/kakao-arena-melon-practice/arena_data/orig/train.json', typ = 'frame')
val = pd.read_json('C:/Users/chees/PycharmProjects/kakao-arena-melon-practice/arena_data/orig/val.json', typ = 'frame')

train_songs = train[['id','songs']]   # id, songs 추출
train_songs_unnest = np.dstack(
    (
        np.repeat(train_songs.id.values, list(map(len, train_songs.songs))),
        np.concatenate(train_songs.songs.values)
    )
)

train_songs_map = pd.DataFrame(data = train_songs_unnest[0], columns = train_songs.columns)
train_songs_map['id'] = train_songs_map['id'].astype(str)
train_songs_map['songs'] = train_songs_map['songs'].astype(str)

train_songs_map_sp = train_songs_map.loc[0:9999]
table = pd.crosstab(train_songs_map_sp.id, train_songs_map_sp.songs)
table = pd.DataFrame(table)

# Association Rule Analysis
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

table_sets = table.applymap(encode_units)
table_sets.drop('POSTAGE', inplace=True, axis=1)

frequent_itemsets = apriori(table, min_support=0.07, use_colnames=False)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()
