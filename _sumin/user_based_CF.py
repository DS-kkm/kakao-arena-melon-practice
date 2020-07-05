"""

### train
- playlist : 92,056 개

### questions
- playlist : 23,015 개

### train + questions
- song : 576,168 개
- tag : 26,586 개
"""

# -*- coding: utf-8 -*-
import fire
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from scipy.sparse import coo_matrix
from arena_util import write_json, remove_seen



class CFUserBased:

    def _most_popular(self, playlists, col, topk_count):
        """arena_util.py의 most_popular 함수 customizing"""
        c = Counter()

        for items in playlists[col]:
            c.update(items)

        topk = c.most_common(topk_count)
        return c, [k for k, _ in topk]

    def _add_new_id(self, df, column):
        data = df[column]
        item_counter = Counter([datum for datums in data for datum in datums])
        item_dict = {x: item_counter[x] for x in item_counter}
        id_newid = dict()
        newid_id = dict()
        for i, x in enumerate(item_dict):
            id_newid[x] = i
            newid_id[i] = x
        len_item = len(item_dict)  # 총 item의 개수
        return id_newid, newid_id, len_item

    def _generate_answers(self, train, questions):

        _, song_mp = self._most_popular(train, "songs", 200)
        _, tag_mp = self._most_popular(train, "tags", 100)

        len_train = len(train)  # 92056 개
        len_question = len(questions)  # 23015 개

        plylst = pd.concat([train, questions])
        plylst["nid"] = range(len_train + len_question)
        # plylst_id_nid = dict(zip(plylst["id"], plylst["nid"]))
        plylst_nid_id = dict(zip(plylst["nid"], plylst["id"]))

        # song에 새로운 id로 sid 부여
        # len_songs : 576168
        song_id_sid, song_sid_id, len_songs = self._add_new_id(plylst, "songs")

        # tag에 id로 tid 부여
        # len_tags : 26586
        tag_id_tid, tag_tid_id, len_tags = self._add_new_id(plylst, "tags")

        plylst['songs_id'] = plylst['songs']\
            .map(lambda x: [song_id_sid.get(s) for s in x
                            if song_id_sid.get(s) is not None])
        plylst['tags_id'] = plylst['tags']\
            .map(lambda x: [tag_id_tid.get(t) for t in x
                            if tag_id_tid.get(t) is not None])

        plylst_use = plylst[['nid', 'songs_id', 'tags_id']].copy()
        plylst_use['num_songs'] = plylst_use['songs_id'].str.len()
        plylst_use['num_tags'] = plylst_use['tags_id'].str.len()
        plylst_use = plylst_use.set_index('nid')

        plylst_train = plylst_use.iloc[:len_train, :]
        plylst_test = plylst_use.iloc[len_train:, :]

        row = np.repeat(range(len_train), plylst_train['num_songs'])  # 4239978
        col = [song for songs in plylst_train['songs_id'] for song in songs]  # 4239978
        dat = np.repeat(1, plylst_train['num_songs'].sum())  # 4239978
        train_songs_A = coo_matrix((dat, (row, col)),
                                   shape=(len_train, len_songs))  # (92056, 576168)
        train_songs_A_T = train_songs_A.T.tocsr()

        row = np.repeat(range(len_train), plylst_train['num_tags'])
        col = [tag for tags in plylst_train['tags_id'] for tag in tags]
        dat = np.repeat(1, plylst_train['num_tags'].sum())
        train_tags_A = coo_matrix((dat, (row, col)),
                                  shape=(len_train, len_tags))
        train_tags_A_T = train_tags_A.T.tocsr()

        # song, tag 추천
        ans = []
        for pid in tqdm(plylst_test.index):

            # 예측할 플레이리스트에 들어있는 song과 tag 확인
            songs_already = plylst_test.loc[pid, "songs_id"]
            tags_already = plylst_test.loc[pid, "tags_id"]

            if not songs_already:
                rec_song_idx = song_mp
                rec_tag_idx = remove_seen(tags_already, tag_mp)
            else:
                p = np.zeros((len_songs, 1))  # (576168, 1)
                p[plylst_test.loc[pid, 'songs_id']] = 1

                val = train_songs_A.dot(p).reshape(-1)  # (92056, )

                cand_song = train_songs_A_T.dot(val)  # (576168, )
                cand_song_idx = cand_song.reshape(-1).argsort()[-200:][::-1]
                cand_song_idx = remove_seen(songs_already, cand_song_idx)
                rec_song_idx = [song_sid_id[i] for i in cand_song_idx]

                cand_tag = train_tags_A_T.dot(val)
                cand_tag_idx = cand_tag.reshape(-1).argsort()[-20:][::-1]
                cand_tag_idx = remove_seen(tags_already, cand_tag_idx)
                rec_tag_idx = [tag_tid_id[i] for i in cand_tag_idx]

            ans.append({
                "id": plylst_nid_id[pid],
                "songs": rec_song_idx[:100],
                "tags": rec_tag_idx[:10]
            })

        return ans

    def run(self, train_fname, question_fname):

        print("Loading train file...")
        train_data = pd.read_json(train_fname)  # arena_data/orig/train.json

        print("Loading question file...")
        questions = pd.read_json(question_fname)  # arena_data/questions/val.json

        print("Generating answers...")
        answers = self._generate_answers(train_data, questions)

        print("Writing answers...")
        write_json(answers, "results/results.json")


if __name__ == "__main__":
    fire.Fire(CFUserBased)
