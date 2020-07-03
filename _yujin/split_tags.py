#train_data에서 태그를 일일이 분해
tags = [i for i in train_data['tags']]
tag_set = set()
for i in tags:
    for tag in i: tag_set.add(tag)
tag2idx = dict()
for idx, tag in enumerate(tag_set) : tag2idx[tag] = tag, idx
col_list = ['id']
col_list += [i for i in tag2idx]
tag_df = pd.DataFrame(columns=col_list) #태그만이 column으로 있는 데이터프레임 완성
