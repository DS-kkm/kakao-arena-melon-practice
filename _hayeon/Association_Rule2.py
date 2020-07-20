"""
    ### Association Rule Mining

    # train : 학습데이터, 92,056개
    # val : 검증데이터, 23,015개

    ### Algorithm

    # title only -> most popular song 추천
      song only / tags only -> song과 tags 기반 추천
      song and tags -> tags 기반 중복되는 song 추천

"""

# -*- coding: utf-8 -*-
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

train = pd.read_json('C:/Users/chees/PycharmProjects/kakao-arena-melon-practice/arena_data/orig/train.json')
val = pd.read_json('C:/Users/chees/PycharmProjects/kakao-arena-melon-practice/arena_data/orig/val.json')

Class ARMplaylist:

    def _



    def run(self, train_fname, test_fname)

        print("Reading train data...\n")
        train_data = pd.read_json(train_fname)

        print("Reading validation data...\n")
        test_data = pd.read_json(test_fname)

        print("Scoring...")
        results = self._(train_data, test_data)

        print("Writing results...")
        write_json(results, "results/results.json")

if __name__ == "__main__":

    fire.Fire(ARMplaylist)