from stopwords import stopwords
from eval_lda import eval_lda
from useModel_v3 import predict, fashion_item_classifier_model, text_pipeline, classifier_category
from making_prompt import making_prompt
import time
import pandas as pd
import gensim
import psycopg2
from gensim import corpora
from io import StringIO
import numpy as np
from DB_connector import get_db_connection
from keras.models import load_model
from mecab import MeCab
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
import os
import urllib
from PIL import Image
from karlo import t2i
from papago import translate_with_papago
mecab1 = MeCab()

# 불용어 뒤에 띄어쓰기 있는것 까지 같이 추가해서 제거
new_stopwords = []
for word in stopwords:
    new_stopwords.append(word + ' ')
    new_stopwords.append(word)
    new_stopwords.append(' ' + word)
    new_stopwords.append(' ' + word + ' ')


# 이제 위 에거 이후 부터는 DB 에서 바로 불러다 쓰자
conn = get_db_connection()
cursor = conn.cursor()

cursor.execute("select comments from video_datas_normalized_1")
rows = cursor.fetchall()

cursor.close()
conn.close()


# 이제 불러다가 쓰기 전에, 중복제거를 좀 해야 함
# set함수를 사용하여 중복제거, 단 순서는 유지 안됨.
# unique_comments = list(set(rows))

# [(['1','2',...],), (['1', '2', ... ],), ]

# 모든 댓글을 풀어서 하나의 리스트에 담기
list = []
for tuple in rows:
    comments_list = tuple[0]
    list.extend(comments_list)


# print(rows[0:2])
unique_comments = set(list)


# # 빈도수 낮은 단어들을 찾기 위한 토큰화 과정

# temp = []
# for sentence in unique_comments:
#     temp_X = mecab1.morphs(sentence[0])

#     temp_X = [word for word in temp_X if not word in new_stopwords]  # 불용어 제거
#     temp.append(temp_X)


# # Tokenizer 정의 및 훈련
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(temp)


# # 단어 빈도수 계산
# threshold = 10
# total_cnt = len(tokenizer.word_index)  # 전체 단어 수
# rare_cnt = 0  # 빈도수가 기준 이하인 단어 수
# rare_freq = 0  # 기준 미만의 단어 빈도수 총합

# for key, value in tokenizer.word_counts.items():
#     if value < threshold:
#         rare_cnt += 1
#         rare_freq += value

# print('전체 단어 수:', total_cnt)
# print('빈도수가 {}번 미만인 희귀 단어 수: {}'.format(threshold, rare_cnt))
# print('희귀 단어 비율: {:.2f}%'.format((rare_cnt / total_cnt) * 100))
# print('희귀 단어 등장 비율: {:.2f}%'.format(
#     (rare_freq / sum(tokenizer.word_counts.values())) * 100))

# # 희귀 단어를 제외한 단어 사전 생성
# vocab_size = total_cnt - rare_cnt


# # 정규화까지 진행된 댓글리스트를 다시 가져와서 0~4까지 분류
# temp_2 = []
# for sentence in unique_comments:
#     temp_X = mecab1.nouns(sentence[0])

#     temp_X = [word for word in temp_X if not word in new_stopwords]  # 불용어 제거
#     temp_2.append(temp_X)


# tokenizer = Tokenizer(num_words=vocab_size)
# tokenizer.fit_on_texts(temp_2)

# # 텍스트 시퀀스를 정수 시퀀스로 변환
# sequences = tokenizer.texts_to_sequences(temp_2)

# # 패딩
# max_sequence_length = 150
# padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)


# # 저장된 모델 로드
# loaded_model = load_model("classfy_model.h5")

# # 댓글에 대한 예측 수행
# predictions = loaded_model.predict(padded_sequences)

# # 임계값을 기준으로 라벨 결정
# predicted_labels = []
# for pred in predictions:
#     if all(prob < 0.5 for prob in pred):
#         predicted_labels.append(5)  # 모든 클래스의 예측 확률이 50% 미만일 경우 라벨을 6으로 설정
#     else:
#         predicted_labels.append(np.argmax(pred))  # 그렇지 않으면 가장 높은 확률의 라벨을 선택

# # 예측 결과를 데이터프레임으로 변환
# classified_comments_df = pd.DataFrame({
#     'comment': unique_comments,
#     'label': predicted_labels
# })


# print(classified_comments_df)


# conn = get_db_connection()
# Pandas DataFrame을 PostgreSQL 테이블에 삽입하는 함수


# def insert_into_table(conn, df, table):
#     """
#     데이터프레임의 데이터를 PostgreSQL 테이블에 INSERT 쿼리를 사용하여 삽입
#     """
#     cursor = conn.cursor()

#     # INSERT 쿼리의 칼럼 부분과 값 부분을 정의
#     # 예를 들어, 데이터프레임에 'label'과 'text'라는 두 개의 칼럼이 있다고 가정
#     columns = ', '.join(df.columns)  # 'label, text'
#     placeholders = ', '.join(['%s'] * len(df.columns))  # '%s, %s'

#     query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

#     for index, row in df.iterrows():
#         try:
#             # row.values는 데이터프레임 행의 모든 값을 포함하는 배열입니다.
#             cursor.execute(query, tuple(row.values))
#             conn.commit()
#         except (Exception, psycopg2.DatabaseError) as error:
#             print(f"Error: {error}")
#             conn.rollback()
#             cursor.close()
#             return 1

#     print("Data inserted successfully")
#     cursor.close()


# 데이터프레임을 PostgreSQL 테이블에 삽입
# insert_into_table(conn, classified_comments_df, 'classified_comments')

# 이제 이것들을 LDA모델에 보내서 키워드를 추출 하자

def lda(comment_list):
    temp = []
    for sentence in comment_list:
        temp_X = mecab1.nouns(sentence)  # 명사만 뽑는다.
        temp_X = [word for word in temp_X if not word in new_stopwords]  # 불용어 제거
        temp.append(temp_X)

    # 단어의 빈도수 계산
    word_freq = {}
    for text in temp:
        for word in text:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    # 임계값 이하의 빈도수를 가진 단어 제거
    # 여기서 TEMP는 하나의 댓글이 토큰화된 리스트 ["안녕", "나는", "성호다"] 가 여러개 뭉쳐 있는 형태이고
    # TEXT는 댓글 하나를 의미한다.
    threshold = 10
    filtered_temp = [
        [word for word in text if word_freq[word] >= threshold] for text in temp]

    # Gensim에 사용할 단어 사전 생성
    gensim_dictionary = corpora.Dictionary(filtered_temp)

    # 문서-단어 매트릭스 생성

    corpus = [gensim_dictionary.doc2bow(text) for text in filtered_temp]
    time1 = time.time()

    # LDA 모델 학습
    # 여기서 하이퍼 파라미터 조절

    # 나중에 평가 지표에 따라서 auto 일때랑 비교.
    '''
    알파값
        시작점: 일반적으로 alpha 값의 좋은 시작점은 1 / 토픽 수입니다. 이는 각 문서가 모든 토픽을 균등하게 포함할 것이라는 기대를 나타냅니다.
        조정 방향:
        값을 높이면: 문서가 더 많은 토픽을 포함하게 됩니다 (토픽 분포가 더 균등해짐).
        값을 낮추면: 문서가 더 적은 수의 토픽에 집중하게 됩니다 (토픽 분포가 더 집중됨).
    베타값
        시작점: 베타값의 좋은 시작점은 0.01 또는 1 / 단어 수입니다. 이는 각 주제가 소수의 단어에 집중할 것이라는 기대를 나타냅니다.
        조정 방향:
        값을 높이면: 주제가 더 많은 단어를 포함하게 됩니다 (단어 분포가 더 균등해짐).
        값을 낮추면: 주제가 더 적은 수의 단어에 집중하게 됩니다 (단어 분포가 더 집중됨).
    토픽수
        시작점: 토픽 수의 적절한 시작점은 데이터셋의 크기와 복잡성에 따라 다릅니다. 소규모 데이터셋에는 5-10개의 토픽, 큰 데이터셋에는 10-30개의 토픽을 시작점으로 사용할 수 있습니다.
        조정 방향:
        값을 높이면: 더 많은 특화된 토픽이 생성됩니다. 이는 데이터에 더 세분화된 패턴을 포착할 수 있게 해줍니다.
        값을 낮추면: 더 일반적인 토픽이 생성됩니다. 이는 데이터의 광범위한 패턴을 포착할 수 있게 해줍니다.
    '''

    num_topics = 7
    lda_model = gensim.models.LdaModel(
        alpha=1/7,
        eta=1/3,
        corpus=corpus, num_topics=num_topics, id2word=gensim_dictionary, passes=30)

    # 학습된 토픽들 출력
    topics = lda_model.print_topics(num_words=3)
    for topic in topics:
        print(topic)

    time2 = time.time()

    print('걸린시간:', time2 - time1)

    words_kr = []
    for idx, topic in topics:
        words_kr.append([word.split('*')[1].replace('"', '').strip()
                        for word in topic.split('+')])

    return (words_kr, gensim_dictionary, corpus, lda_model, filtered_temp)


def making_img(prompt):
    # words_en = translate_with_papago(words_kr)
    response = t2i(prompt)
    result = Image.open(urllib.request.urlopen(
        response.get("images")[0].get("image")))
    result.show()


# DB에서 긍정적인 댓글로 분류된 댓글들만 가져와서
# lda를 통해 패션 키워드만 뽑는다.
args = lda(unique_comments)

fashion_keywords = args[0]
# args[0] 이 패션 아이템으로 추출된 키워드들이다.
# lda[1] 은 평가지표를 사용하기 위한 lda 사전이다.  gensim_dictionary
# lda[2] 은 평가지표를 사용하기 위한 lda 사전이다.  corpus
# lda[3] 은 평가지표를 사용하기 위한 lda 사전이다.  lda_modal
# lda[4] 은 평가지표를 사용하기 위한 lda 사전이다.  이게 text

print('이 아래 확인')
# print(args)


if __name__ == '__main__':
    # 여기에서 eval_lda 호출
    eval_lda(texts=args[4], corpus=args[2],
             lda_model=args[3], dictionary=args[1])

# for topic in fashion_keywords:
#     predicted_class = predict(
#         topic, fashion_item_classifier_model, text_pipeline)
#     # classifier_category 함수를 통해 패션 아이템을 카테고리별로 분류한다. 그리고 분류된 패션 아이템을 gpt로 보낸다.
#     top_keywords, bottom_keywords, outer_keywords, shoes_keywords, accessary_keywords, not_fashion_items = classifier_category(
#         predicted_class)

#     print('다음 아래 확인해')
#     print(top_keywords, bottom_keywords, outer_keywords,
#           shoes_keywords, accessary_keywords, not_fashion_items)

#     # call_cnt = 5
#     # for num in range(call_cnt):
#     styling_prompt = making_prompt(top_keywords, bottom_keywords,
#                                    outer_keywords, shoes_keywords, accessary_keywords)

#     print('프롬프트: ', styling_prompt)
#     making_img(styling_prompt)


# # 하이퍼파라미터조절 (최적화)
'''
# 1. 문서-토픽 분포
# alpha
# 토픽이 얼마나 균일하게 분포...하는지 -> 낮은 알파값은 몇몇 토픽에만 집중하게함.
# 2. 토픽-단어 분포
# eta
# 수치가 높을수록 다양한 키워드에 집중. 낮은수록 특정단어에 더 집중
# 3. 토픽 수 조정


'''
