import fire
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from arena_util import write_json
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype


class Recommend:
    def run(self):
        song_meta = pd.read_json('./res/song_meta.json')

        print("Loading train file...")
        train_data = pd.read_json('./arena_data/orig/train.json', encoding='utf-8')

        print("Loading question file...")
        question_data = pd.read_json('./arena_data/questions/val.json',encoding='utf-8')

        print("Generating answers...")
        answers = self.related(train_data,question_data)

        print("Writing answers...")
        write_json(answers, 'results/results.json')

    def related(self,train_data,question_data):
        id_songs = (
            train_data[['id', 'songs']]
           .explode('songs')
           .assign(value=1)
           .rename(columns={'id': 'user_id', 'songs': 'item_id'})
            )

        id_tags = (
            train_data[['id', 'tags']]
                .explode('tags')
                .assign(value=1)
                .rename(columns={'id': 'user_id', 'tags': 'tag'})
        )

        id_songs['user_id'] = id_songs['user_id'].astype(np.int32)
        id_songs['item_id'] = id_songs['item_id'].astype(np.int32)
        id_songs['value'] = id_songs['value'].astype(np.int8)
        id_tags['user_id'] = id_tags['user_id'].astype(np.int32)
        id_tags['value'] = id_tags['value'].astype(np.int8)

        while True:
            prev = len(id_songs)

            # 5곡 이상 가진 플레이 리스트만
            user_count = id_songs.user_id.value_counts()
            id_songs = id_songs[id_songs.user_id.isin(user_count[user_count >= 5].index)]

            # 5번 이상 등장한 곡들만
            item_count = id_songs.item_id.value_counts()
            id_songs = id_songs[id_songs.item_id.isin(item_count[item_count >= 5].index)]

            cur = len(id_songs)

            if prev == cur: break

        while True:
            prev = len(id_tags)

            # 5태그 이상 가진 플레이 리스트만
            user_count = id_tags.user_id.value_counts()
            id_tags = id_tags[id_tags.user_id.isin(user_count[user_count >= 5].index)]

            # 5번 이상 등장한 태그들만
            tag_count = id_tags.tag.value_counts()
            id_tags = id_tags[id_tags.tag.isin(tag_count[tag_count >= 5].index)]

            cur = len(id_tags)

            if prev == cur: break

        person_c_item = CategoricalDtype(sorted(id_songs.user_id.unique()), ordered=True)
        thing_c_item = CategoricalDtype(sorted(id_songs.item_id.unique()), ordered=True)
        person_c_tag = CategoricalDtype(sorted(id_tags.user_id.unique()), ordered=True)
        thing_c_tag = CategoricalDtype(sorted(id_tags.tag.unique()), ordered=True)

        row_item = id_songs.user_id.astype(person_c_item).cat.codes
        col_item = id_songs.item_id.astype(thing_c_item).cat.codes
        item_matrix = csr_matrix((id_songs["value"], (row_item, col_item)), \
                                   shape=(person_c_item.categories.size, thing_c_item.categories.size))

        row_tag = id_tags.user_id.astype(person_c_tag).cat.codes
        col_tag = id_tags.tag.astype(thing_c_tag).cat.codes
        tag_matrix = csr_matrix((id_tags["value"], (row_tag, col_tag)), \
                                shape=(person_c_tag.categories.size, thing_c_tag.categories.size))

        song_idx2id = {i: j for i, j in enumerate(thing_c_item.categories)}
        song_id2idx = {j: i for i, j in enumerate(thing_c_item.categories)}
        tag_idx2name = {i: j for i, j in enumerate(thing_c_tag.categories)}
        tag_name2idx = {j: i for i, j in enumerate(thing_c_tag.categories)}

        import implicit

        # initialize a model
        item_model = implicit.als.AlternatingLeastSquares(factors=50)
        tag_model = implicit.als.AlternatingLeastSquares(factors=50)

        # train the model on a sparse matrix of item/user/confidence weights
        item_model.fit(item_matrix.T)
        tag_model.fit(tag_matrix.T)

        # recommend items for a user
        user_items = item_matrix.tocsr()
        user_tags = tag_matrix.tocsr()

        # 인기곡
        top200_songs = id_songs.groupby('item_id').value.sum().nlargest(200)
        top200_tags = id_tags.groupby('tag').value.sum().nlargest(200)

        question_ids = question_data.id.tolist()

        question_item = (
            question_data[['id', 'songs']]
            .explode('songs')
            .assign(value=1)
            .rename(columns={'id': 'user_id', 'songs': 'item_id'})
            )
        question_tag = (
            question_data[['id', 'tags']]
            .explode('tags')
            .assign(value=1)
            .rename(columns={'id': 'user_id', 'tags': 'tag'})
            )

        # 각 플레이리스트에 대해
        res = {}
        for idx, row in tqdm(question_data.iterrows(), total=len(question_data)):
            cands = {}
            # 각 곡 별로
            for song in row.songs:
                # 유사곡 100개 소환.
                if song not in song_id2idx.keys(): continue  # val에는 train에 아예 없던 곡이 나올 수 있으므로 해당하면 재끼도록 설계
                related = [(song_idx2id[r[0]], r[1]) for r in item_model.similar_items(song_id2idx[song], 100)]
                for cand in related:
                    # 추천된 아이템 : 유사도 리스트를 딕셔너리로 구현
                    cands[cand[0]] = cands.get(cand[0], []) + [cand[1]]
            # 한 곡이 여러번 추천될 땐 가장 높은 유사도 하나 채택
            cands = {k: max(v) for k, v in cands.items()}
            # 유사도 순으로 정렬 후, 기존 플레이리스트에 없는 곡들만 가지고 100개 뽑기
            sorted_cands = [w for w in sorted(cands, key=cands.get, reverse=True) if w not in row.songs][:100]

            # 가끔 미쳐가지고 비어있거나 한 경우도 있음. 이럴 땐 그냥 베스트를 넣어주자
            if len(sorted_cands) < 100:
                non_seen_top_200_songs = [song for song in top200_songs if song not in row.songs]
                sorted_cands += non_seen_top_200_songs[:100 - len(sorted_cands)]

            sorted_cands = [int(s) for s in sorted_cands]
            res[row.id] = sorted_cands

        # 각 플레이리스트에 대해
        tag_rec = {}
        for idx, row in tqdm(question_data.iterrows(), total=len(question_data)):
            cands = {}
            # 각 태그 별로
            for tag in row.tags:
                # 유사곡 100개 소환.
                if tag not in tag_name2idx.keys(): continue  # val에는 train에 아예 없던 곡이 나올 수 있으므로 해당하면 재끼도록 설계
                related = [(tag_idx2name[r[0]], r[1]) for r in tag_model.similar_items(tag_name2idx[tag], 10)]
                for cand in related:
                    # 추천된 아이템 : 유사도 리스트를 딕셔너리로 구현
                    cands[cand[0]] = cands.get(cand[0], []) + [cand[1]]
            # 한 곡이 여러번 추천될 땐 가장 높은 유사도 하나 채택
            cands = {k: max(v) for k, v in cands.items()}
            # 유사도 순으로 정렬 후, 기존 플레이리스트에 없는 곳들만 가지고 100개 뽑기
            sorted_cands = [w for w in sorted(cands, key=cands.get, reverse=True) if w not in row.tags][:10]

            # 가끔 미쳐가지고 비어있거나 한 경우도 있음. 이럴 땐 그냥 베스트를 넣어주자
            if len(sorted_cands) < 10:
                non_seen_top_200_tags = [tag for tag in top200_tags.index if tag not in row.tags]
                sorted_cands += non_seen_top_200_tags[:10 - len(sorted_cands)]

            assert len(sorted_cands) == 10

            tag_rec[row.id] = sorted_cands

        answer = []
        for _id, i, j in zip(question_ids, res, tag_rec):
            answer.append({
                "id": _id,
                "songs": res[i][:100],
                "tags": tag_rec[j][:10]
            })

        return answer

if __name__ == "__main__":
    fire.Fire(Recommend)