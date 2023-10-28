import squarify
from collections import Counter
import re
from mecab import MeCab
import urllib.request
import matplotlib as mpl
import matplotlib.pyplot as plt
# 자연어 처리
# 모델 만들고
# 데이터 삽입 계속 확인해보고 데이터 모아야함
# 자동으로 계속 업데이트
#


# 한국어 형태소 분석기
# curl -s https://raw.githubusercontent.com/teddylee777/machine-learning/master/99-Misc/01-Colab/mecab-colab.sh | bash
# pip install koreanize-matplotlib
# pip install python-mecab-ko


# url = 'https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt'
# raw = urllib.request.urlopen(url).read().decode('utf-8')
raw = urllib.request.urlopen(
    'https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt').read()
print(raw[:100])

# 데이터를 문자열로 변환
# 리뷰만 추출
# raw = [if not type(x) x.decode() for x in raw[2:]]
raw_lines = raw.splitlines()
decoded_lines = [line.decode('utf-8') for line in raw_lines]

# for i in raw[2:]:
#   print(i.decode())

# raw = [x.decode() for x in raw[2:]]

reviews = []
for i in decoded_lines:
    reviews.append(i.split('\t')[1])

print(reviews[:5])


# 토큰화
tagger = MeCab()

reviews_nouns = []
for review in reviews:
    for noun in tagger.nouns(review):
        reviews_nouns.append(noun)

reviews_nouns[:20]


# 불용어 리스트 생성
stop_words = '때 그 끝 저 영화 난 나 전 너 넌 당신 뭐 걸 만 이거 저거 중 점 후 뿐 이 저 듯 게 건 것 씨 분 년 게 수 내 이건 이게 편 애 이상 속 데 뭔지 이걸 뭔가 거'
stop_words = stop_words.split(' ')
print(stop_words)

# 필터링
reviews_nouns = []

for review in reviews:
    for noun in tagger.nouns(review):
        if noun not in stop_words:
            reviews_nouns.append(noun)

reviews_nouns[:10]

# 특수문자 제거


def removeSpecialChar(text):
    return re.sub(r'[^\w\s]', '', text)


# 단어 빈도수 확인

reviews_nouns_counter = Counter(reviews_nouns)
top_reviews_nouns = dict(reviews_nouns_counter.most_common(100))
top_reviews_nouns


# 트리맵으로 빈도수 시각화

# 글씨 사이즈 줄이기
plt.rcParams['font.size'] = 8

norm = mpl.colors.Normalize(vmin=min(top_reviews_nouns.values()),
                            vmax=max(top_reviews_nouns.values()))

colors = [mpl.cm.Reds(norm(value)) for value in top_reviews_nouns.values()]

squarify.plot(label=top_reviews_nouns.keys(),
              sizes=top_reviews_nouns.values(),
              color=colors, alpha=0.5)

'''

2. 토큰화 (Tokenization)
텍스트를 단어, 문장, 문단 등의 토큰으로 분리합니다.
예: "I love NLP." → ["I", "love", "NLP", "."]

4. 불용어 제거 (Stopwords Removal)
the, and, is와 같은 빈번하게 등장하지만, 실질적인 의미가 크게 없는 단어를 제거합니다.

5. 특수 문자 및 숫자 제거
분석의 목적에 따라 특수 문자, 숫자, 특정 문자열 등을 제거할 수 있습니다.


6. 어간 추출 및 표제어 추출 (Stemming & Lemmatization)
단어를 그 기본 형태로 변환합니다.
어간 추출: "running" → "run"
표제어 추출: "flies" → "fly"

8. 개체명 인식 (Named Entity Recognition, NER)
텍스트에서 사람, 장소, 조직, 날짜 등의 개체명을 인식합니다.

10. 벡터화 (Vectorization)
텍스트 데이터를 수치 벡터 형태로 변환합니다.

7. 품사 태깅 (POS Tagging)
단어의 품사를 판별하여 태깅합니다.
예: "They refuse to permit us to obtain the refuse permit." → [("They", "PRP"), ("refuse", "VBP"), ...]

9. n-gram 추출
연속된 n개의 토큰으로 구성된 구문을 추출합니다.
예: "I love NLP"의 바이그램 → ["I love", "love NLP"]
One-hot Encoding, Count Vectorizer, TF-IDF, Word2Vec, FastText, BERT Embedding 등



11. 패딩 (Padding)
동일한 길이의 입력 데이터가 필요한 모델에 사용되는 경우, 짧은 문장들을 일정한 길이로 맞춰주는 과정입니다.
12. 특징 추출 (Feature Extraction)
텍스트 데이터에서 통계적, 구조적 특징을 추출합니다.
13. 데이터 분할 (Splitting)
주어진 데이터를 학습, 검증, 테스트 세트로 분할합니다.


'''
