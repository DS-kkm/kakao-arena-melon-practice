"""
    ### Association Rule Mining

    # train : 학습데이터, 92,056개
    # val : 검증데이터, 23,015개

    ### Algorithm

    # title only => most popular song 추천
      song only / tags only => song과 tags 기반 추천
      song and tags => tags 기반 중복되는 song 제외하고 추천

"""

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm

from genre_most_popular import genre_most_popular


train = pd.read_json('C:/Users/chees/PycharmProjects/kakao-arena-melon-practice/arena_data/orig/train.json')
val = pd.read_json('C:/Users/chees/PycharmProjects/kakao-arena-melon-practice/arena_data/orig/val.json')

Class ARMplaylist:

    def _generate_answers(self, train, questions):



    def run(self, train_fname, question_fname)

        print("Loading train file...\n")
        train_data = pd.read_json(train_fname)

        print("Loading question file...\n")
        questions = pd.read_json(question_fname)

        print("Writing answers...\n")
        answer = self.generate_answers(train_data, questions)
        write_json(answers, "results/results.json")

if __name__ == "__main__":

    fire.Fire(ARMplaylist)