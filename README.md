# kakao-arena-melon-practice
대회 연습용 repository

<br>

## 환경 설정
```shell script
conda env create -f environment.yaml
```

<br>

## 데이터
* [멜론 플레이리스트 데이터 탐색](https://brunch.co.kr/@kakao-it/343) 정독!

### genre_gn_all.json
* `장르 코드`, `장르` 쌍
* 대분류 장르 : 30개
* 상세 장르 : 224개

### song_meta.json
* 707,989 곡의 `발매일자`, `대분류 장르`, `상세 장르`, `아티스트명`, `앨범명`

### train.json
* 115,071 개의 플레이리스트의 `태그`, `수록곡`, `좋아요 횟수`, `업데이트 일시 정보`
* 곡 수 : 615,142개 (전체 메타곡의 86.9%)
* 유니크 태그 수 : 29,160개


<br>

## 실행

### (1) 데이터 분할
```shell script
python split_data.py run res/train.json
```

<br>

### (2) 추천 결과 생성
* 추천 결과는 `arena_data/results/results.json` 에 저장됨
* 해당 `results.json` 파일과 소스코드를 각각 zip파일로 압축한 후에 아레나 홈페이지로 제출하시면 점수를 확인 가능

* `most_popular.py
` : 주어진 전체 플레이리스트에서 가장 많이 등장한 곡과 태그를 모든 문제에 대해서 답안으로 내놓는 모델
	
	```bash
	$> python most_popular.py run \
		--train_fname=arena_data/orig/train.json \
		--question_fname=arena_data/questions/val.json 
	```

* `genre_most_popular.py` : 주어지는 각 문제마다, 가장 많이 등장하는 장르에 대해 해당 장르에서 가장 빈번하게 등장하는 곡들을 답안으로 내놓는 모델
    * 위의 모델보다 성능이 약간 향상된 것을 확인할 수 있음
    	
	```bash
	$> python genre_most_popular.py run \
		--song_meta_fname=res/song_meta.json \
		--train_fname=arena_data/orig/train.json \
		--question_fname=arena_data/questions/val.json 
	```
 
<br>

### (3) 평가
* 아레나에 제출하기 전에 본인이 만든 문제 / 정답 세트로 점수를 알아볼 수 있음
* ※ 주의 ※
    * `questions/val.json`을 패러미터로 넣었던 위의 추천 결과 생성 스크립트와는 다르게, `answers/val.json
`을 패러미터로 넣어야 함

```bash
$> python evaluate.py evaluate \
	--gt_fname=arena_data/answers/val.json \
	--rec_fname=arena_data/results/results.json 
```