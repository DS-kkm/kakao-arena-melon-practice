import fire
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from scipy.sparse import coo_matrix
from arena_util import write_json, remove_seen
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype


class Recommend:
    def run(self, train_data, question_data):
        song_meta = pd.read_json('./res/song_meta.json', encoding='utf-8')

        print("Loading train file...")
        train_data = pd.read_json('./arena_data/orig/train.json', encoding='utf-8')

        print("Loading question file...")
        question_data = pd.read_json('./arena_data/questions/val.json',encoding='utf-8')

        print("Generating answers...")
        answers = self.related(train_data,question_data)

        print("Writing answers...")
        write_json(answers, 'results/results.json')

    def related(self,song_meta,train_data,question_data):
        _, song_mp = self.related_song(question_data)
        _, tag_mp = self.related_tag(train_data,100)

        def related_song(question_data):
            song_list = song_meta[['id']]
            id_list = train_data[['id']]

            song_meta_ = song_meta['id'].tolist()
            songs = [i for i in train_data['songs']]
            id_train_ = [i for i in train_data['id']]
            recommend_song = question_data['songs'].tolist()

            dic = {}
            for idx, row in tqdm(train_data.iterrows(), total=len(train_data)):
                dic[row.id] = row.songs

            id_songs = (
                train_data[['id', 'songs']]
                    .explode('songs')
                    .assign(value=1)
                    .rename(columns={'id': 'user_id', 'songs': 'item_id'})
            )

            assert id_songs['user_id'].max() < 2147483647
            assert id_songs['item_id'].max() < 2147483647

            id_songs['user_id'] = id_songs['user_id'].astype(np.int32)
            id_songs['item_id'] = id_songs['item_id'].astype(np.int32)
            id_songs['value'] = id_songs['value'].astype(np.int8)

            # 행렬이 너무 커서 빈도수 적은 요인들 제거
            id_songs.user_id.nunique() * id_songs.item_id.nunique()

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

            id_songs.user_id.nunique() * id_songs.item_id.nunique()  # 10%정도로 감소함

            person_c = CategoricalDtype(sorted(id_songs.user_id.unique()), ordered=True)
            thing_c = CategoricalDtype(sorted(id_songs.item_id.unique()), ordered=True)

            row = id_songs.user_id.astype(person_c).cat.codes
            col = id_songs.item_id.astype(thing_c).cat.codes
            sparse_matrix = csr_matrix((id_songs["value"], (row, col)), \
                                       shape=(person_c.categories.size, thing_c.categories.size))

            import implicit

            # initialize a model
            model = implicit.als.AlternatingLeastSquares(factors=50)

            # train the model on a sparse matrix of item/user/confidence weights
            model.fit(sparse_matrix.T)

            # recommend items for a user
            user_items = sparse_matrix.tocsr()

            # find related items
            related = model.similar_items(recommend_song[])



