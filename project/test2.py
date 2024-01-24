
# lda 함수
from gensim import corpora
import gensim


# 샘플 문장
sample_sentence = "The quick brown fox jumps over the lazy dog."

# 단계 1: 문장 토큰화
tokens = sample_sentence.lower().split()

# 단계 2: 사전 생성
dictionary = corpora.Dictionary([tokens])

# 단계 3: 문서-단어 행렬 생성
bow_corpus = [dictionary.doc2bow(tokens)]

# 단계 4: LDA 모델 훈련
lda_model = gensim.models.ldamodel.LdaModel(
    bow_corpus, num_topics=1, id2word=dictionary, passes=15)

# 단계 5: 결과 출력
topics = lda_model.print_topics(num_words=4)
for topic in topics:
    print(topic)
