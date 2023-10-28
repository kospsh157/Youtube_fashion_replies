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
패션_아이템_리스트 = [
    "티셔츠", "청바지", "블라우스", "스커트", "셔츠", "슬랙스", "니트", "자켓", "코트", "맨투맨",
    "후드티", "블레이저", "원피스", "점프수트", "바람막이", "패딩", "스웨터", "카디건", "조끼", "스판덱스",
    "트렌치코트", "사파리자켓", "레깅스", "반팔티", "긴팔티", "크롭톱", "탱크탑", "풀오버", "반바지", "롱스커트",
    "미니스커트", "플리츠스커트", "에이라인 스커트", "레더자켓", "데님자켓", "피코트", "카프리팬츠", "코듀로이 팬츠", "플래어 스커트", "보울러 모자",
    "스트라이프 티셔츠", "머메이드 스커트", "세일러 칼라", "볼룬 슬리브", "펜슬 스커트", "맥시드레스", "케이프", "포켓 티셔츠", "드롭 크로치 팬츠", "벨뷸트럼",
    "주름 칼라", "버튼 다운 셔츠", "피쉬테일 스커트", "핀 스트라이프 슬랙스", "볼룬 드레스", "탑코트", "워시드 청바지", "모토 자켓", "오버올", "플로럴 프린트 스커트",
    "체크 셔츠", "오버사이즈 셔츠", "페이즐리 프린트", "셔링 블라우스", "스터드 장식 자켓", "볼륨 스커트", "플로럴 블라우스", "반다나 프린트", "벨트 드레스", "패널 스커트",
    "버티컬 스트라이프 셔츠", "스팽글 드레스", "프린지 장식 코트", "버튼 프론트 스커트", "페플럼 탑", "포켓 드레스", "바이커 자켓", "아노락", "배색 패턴 티셔츠", "오버사이즈 드레스",
    "퍼프 슬리브", "카모플라주 팬츠", "폴카닷 블라우스", "스카시", "테이퍼드 팬츠", "애시메트릭 스커트", "라이닝 장식 자켓", "셔츠 드레스", "오버사이즈 니트", "발레리나 스커트",
    "라이더 자켓", "폴카닷 원피스", "스트라이프 팬츠", "비즈 장식 블라우스", "플리츠 팬츠", "체크 스커트", "라운드 칼라 셔츠", "하이웨이스트 팬츠", "데님 스커트", "벨 슬리브 블라우스",
    "보이프렌드 청바지", "포피츠 원피스", "린넨 팬츠", "코튼 셔츠", "코르셋 원피스", "레이스 블라우스", "오버롤", "버티컬 스트라이프 팬츠", "오버사이즈 코트", "코튼 스커트",
    "페이즐리 셔츠", "프린지 장식 스커트", "스트라이프 니트", "피쉬넷 탑", "체크 팬츠", "페플럼 재킷", "플리츠 원피스", "크로스 프론트 블라우스", "레이스 원피스", "볼륨 팬츠",
    "오버사이즈 블라우스", "프린지 장식 니트", "밴딩 팬츠", "티어드 스커트", "테이퍼드 청바지", "시스루 탑", "패턴 원피스", "스팽글 자켓", "데님 셔츠", "코튼 원피스",
    "포피츠 블라우스", "스톤 워싱 청바지", "오버롤 드레스", "오버사이즈 티셔츠", "라이닝 장식 코트", "플로럴 프린트 팬츠", "플리츠 블라우스", "스트라이프 원피스", "페이즐리 원피스", "포피츠 셔츠"
]
