import gensim
from gensim import corpora
from pprint import pprint

# 댓글 데이터 예시 (이미 전처리된 문장들의 리스트)
documents = [
    "fashion trend this year",
    "love the design of this dress",
    "movie was really touching",
    "best actor performance",
    # ... 여러 댓글들
]

# 토큰화
texts = [doc.split() for doc in documents]

# 단어 사전 생성
dictionary = corpora.Dictionary(texts)

# 문서-단어 매트릭스 생성
corpus = [dictionary.doc2bow(text) for text in texts]

# LDA 모델 학습
num_topics = 2  # 예시로 2개의 토픽을 가정
lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

# 학습된 토픽들 출력
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)

# 새로운 문서에 대한 토픽 분포 예측
new_doc = "new fashion trend in 2023"
new_doc_bow = dictionary.doc2bow(new_doc.split())
print(lda_model.get_document_topics(new_doc_bow))
