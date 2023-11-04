from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from mecab import MeCab
# from gensim import corpora
from pprint import pprint
# from mecab import MeCab # 이 줄은 필요 없으므로 제거합니다. 이미 위에서 `from konlpy.tag import Mecab`을 사용했습니다.
from DB_retriever import select_normalized
import pandas as pd
import requests
import json
import urllib
from PIL import Image
# from karlo import t2i # 이 코드가 실제로 어떤 기능을 하는지에 대한 정보가 없으므로 주석 처리합니다.
# from papago import translate_with_papago # 이 코드 또한 실제 기능에 대한 정보가 없으므로 주석 처리합니다.

# 띄어쓰기 정형화 된 것들 다시 불러와서 데이터프레임으로 저장
rows = select_normalized(['title', 'comments'])
df = pd.DataFrame(rows, columns=['Title', 'Comments'])

# Mecab 토크나이저 초기화
mecab = Mecab()

# 한국어 불용어 리스트 예시
stop_words = ['는', '에', '와', '을', '다', '의',
              '가', '이', '은', '들', '를', '으로', '한', '하다']

# 샘플 문서들
documents = [
    "빠른 갈색 여우가 게으른 개를 뛰어넘다.",
    "개.",
    "여우"
]

# 토큰화를 위한 함수


def tokenize(text):
    tokens = mecab.nouns(text)
    return [token for token in tokens if token not in stop_words]


# TF-IDF 변환기를 초기화합니다. 불용어 처리는 토큰화 함수 내에서 수행합니다.
vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False)

# 문서를 TF-IDF 행렬로 변환합니다.
tfidf_matrix = vectorizer.fit_transform(documents)

# 각 단어와 해당 단어의 인덱스를 출력합니다.
print(vectorizer.vocabulary_)

# 각 문서의 TF-IDF 벡터를 출력합니다.
print(tfidf_matrix.toarray())
