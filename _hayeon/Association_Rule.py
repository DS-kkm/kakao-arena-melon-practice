"""
    ### Association Rule Mining

    # train : 학습데이터, 92,056개
      val : 검증데이터, 23,015개

    # title only : 1,151개
      song only : 6,904개
      tags only : 3,452개
      song and tags : 11,508개
"""

import copy
import numpy as np
import pandas as pd
import itertools

from tqdm import tqdm

from split_data import ArenaSplitter

train = pd.read_json('C:/Users/chees/PycharmProjects/kakao-arena-melon-practice/arena_data/orig/train.json')
val = pd.read_json('C:/Users/chees/PycharmProjects/kakao-arena-melon-practice/arena_data/orig/val.json')

# id & songs 데이터프레임 만들기
plylst_song = train[['id','songs']]

# plylst_song에서 songs 중첩 제거하기
plylst_song_unnest = np.dstack(
    (
        np.repeat(plylst_song.id.values, list(map(len, plylst_song.songs))),
        np.concatenate(plylst_song.songs.values)
    )
)

# unnested 데이터프레임 생성 : plylst_song_map
plylst_song_map = pd.DataFrame(data = plylst_song_unnest[0], columns = plylst_song.columns)
plylst_song_map['id'] = plylst_song_map['id'].astype(str)
plylst_song_map['songs'] = plylst_song_map['songs'].astype(str)

# 빈도테이블 생성
plylst_song_map_str = plylst_song_map.loc[0:9999]
plylst_song_map_str = plylst_song_map_str.sample(n=10000)
table = pd.crosstab(plylst_song_map_str.id, plylst_song_map_str.songs)

# Association Rule 생성
def support(df, item_lst):
    return (df[list(item_lst)].sum(axis=1)==len(item_lst)).mean()

def make_all_set_over_support(df, support_threshold):
    items = []
    single_items = [col for col in df.columns if support(df, [col]) > support_threshold]  # size 1 items

    size = 2
    while True:
        new_items = []
        for item_cand in itertools.combinations(single_items, size):
            # print(item_cand, (df[list(item_cand)].sum(axis=1)==size).mean())
            if support(df, list(item_cand)) > support_threshold:
                new_items.append(list(item_cand))
        if len(new_items) == 0:
            break
        else:
            items += new_items
            size += 1
    items += [[s] for s in single_items]  # 이렇게 해줘야 모든 type이 list가 됨
    return items


def make_confidence_lst(df, item_set_over_support, confidence_threshold):
    r_lst = []
    for item1 in item_set_over_support:
        for item2 in item_set_over_support:
            if len(set(item1).intersection(set(item2))) == 0:
                conf = support(df, list(set(item1).union(set(item2)))) / support(df, item1)
                if conf > confidence_threshold:
                    r_lst.append((item1, item2, conf))
            else:
                continue
    return sorted(r_lst, key=lambda x: x[2], reverse=True)


def make_lift_lst(df, item_set_over_support, lift_threhsold):
    r_lst = []
    for item1 in item_set_over_support:
        for item2 in item_set_over_support:
            if len(set(item1).intersection(set(item2))) == 0:
                lift = support(df, list(set(item1).union(set(item2))))
                lift /= support(df, item1)
                lift /= support(df, item2)
                if lift > lift_threhsold:
                    r_lst.append((item1, item2, lift))
            else:
                continue
    return sorted(r_lst, key=lambda x: x[2], reverse=True)

over_support_lst = make_all_set_over_support(table, 0.07)

for a, b, conf in  make_confidence_lst(table, over_support_lst, 0.53):
    print("{} => {}: {}".format(a, b, conf))

for a, b, lift in  make_lift_lst(df, over_support_lst, 5.6):
    print("{} => {}: {}".format(a, b, lift))