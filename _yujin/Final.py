import fire
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from arena_util import write_json
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype


class Recommend:
    def run(self, train_data, question_data):

        print("Loading train file...")
        train_data = pd.read_json('./arena_data/orig/train.json', encoding='utf-8')

        print("Loading question file...")
        question_data = pd.read_json('./arena_data/questions/val.json',encoding='utf-8')

        print("Generating answers...")
        answers = self.related(train_data,question_data)

        print("Writing answers...")
        write_json(answers, 'results/results.json')

    def related(self,question_data):
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

        assert id_songs['user_id'].max() < 2147483647
        assert id_songs['item_id'].max() < 2147483647
        assert id_tags['user_id'].max() < 2147483647
        assert id_tags['tag'].max() < 2147483647

        id_songs['user_id'] = id_songs['user_id'].astype(np.int32)
        id_songs['item_id'] = id_songs['item_id'].astype(np.int32)
        id_songs['value'] = id_songs['value'].astype(np.int8)
        id_tags['user_id'] = id_tags['user_id'].astype(np.int32)
        id_tags['tag'] = id_tags['tag'].astype(np.int32)
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

        dic_item = {}
        for idx, row in tqdm(question_data.iterrows(), total=len(question_data)):
            dic_item[row.id] = row.songs
        dic_tag = {}
        for idx, row in tqdm(question_data.iterrows(), total=len(question_data)):
            dic_tag[row.id] = row.tags

        set_question_item = set(question_item.user_id)
        ls_question_item = list(set_question_item)
        set_question_tag = set(question_tag.user_id)
        ls_question_tag = list(set_question_tag)

        related = []
        for i in range(len(ls_question_item)):
            for j in range(len(dic_item[ls_question_item[i]])):
                rel = item_model.similar_items(dic_item[ls_question_item[i]][j])
                related.append(rel)

        tag_related = []
        for i in range(len(ls_question_tag)):
            for j in range(len(dic_tag[ls_question_tag[i]])):
                rel = tag_model.similar_items(dic_tag[ls_question_tag[i]][j])
                related.append(rel)

        answer = []
        for i in range(len(ls_question_item)):
            answer.append({
                "id": ls_question_item[i],
                "songs": related[i][:100],
                "tags": tag_related[i][:10]
            })

        return answer