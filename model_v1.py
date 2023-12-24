from DB_connector import get_db_connection
from io import StringIO
import psycopg2
from keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
import pandas as pd
import time
from DB_retriever import select_query
import re
import numpy as np
from pykospacing.kospacing import Spacing
from mecab import MeCab
from stopwords import stopwords

# 불용어 뒤에 띄어쓰기 있는것 까지 같이 추가해서 제거
new_stopwords = []
for word in stopwords:
    new_stopwords.append(word + ' ')
    new_stopwords.append(word)
    new_stopwords.append(' ' + word)
    new_stopwords.append(' ' + word + ' ')
rows = select_query(['title', 'comments'])
# 댓글은 문자열 형태이다...
# 제목도 문자열 형태이다.
# 영상당 하나는 튜플 형태이다. 그러니깐 튜플 하나에 문자열 형태 2개의 원소가 존재한다.

# # DB에 저장되어 있는 데이터 가져다가 딕셔너리 형태로 만들기
# data_list = []
# for tuple in rows:
#     title = tuple[0]
#     comment_list = tuple[1].replace("{", "").replace("}", "").split('","')
#     comment_list[0] = comment_list[0].replace('"', '')

#     dic = {}
#     dic['title'] = title
#     dic['comments'] = comment_list

#     data_list.append(dic)


# # <<<<<<<<<클렌징>>>>>>>>>>>>
# # 1. 특수문자, 외국어 제거


# # 정규식을 이용해서 한글, 공백(띄어쓰기 한칸)를 제외하고 모두 제거하기
# def clean_text(text):
#     cleaned_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ ]", "", text)
#     cleaned_text = re.sub("\s+", " ", cleaned_text)  # 연속된 공백을 하나로 축소
#     return cleaned_text

# # 위 정규식을 거치고 나면 텅텅 비어 있는 값 '', 혹은 공백만 있는 값들이 있을 수 있음
# # 이 값들도 제거하는 정규식 => '^\s*$'


# # 반복문으로 데이터 정리
# for data in data_list:
#     data['title'] = clean_text(data['title'])
#     if re.match(r'^\s*$', data['title']):
#         data['title'] = ''

#     for i, comment in enumerate(data['comments']):
#         data['comments'][i] = clean_text(comment).strip()

# # print(data_list[-1]['comments'][0])


# comments_list = []
# for one_video in data_list:
#     for comments_of_one in one_video['comments']:
#         # print(comments_of_one)
#         if comments_of_one != '':
#             comments_list.append(comments_of_one)


# time3 = time.time()

# # 띄어쓰기 정규화하기
# spacing = Spacing()
# normarlized_comments = []
# for comment in comments_list:
#     normal_str = spacing(comment)
#     print(normal_str)
#     normarlized_comments.append(normal_str)

# time4 = time.time()
# print('띄어쓰기 정규화 하는데 걸린 시간:', time4 - time3)


# # 위에서 정규화한 리스트들 파일형태로 저장
# # 파일로 저장
# with open("output.txt", "w", encoding="utf-8") as file:
#     for string in normarlized_comments:
#         file.write(string + "\n")  # 각 문자열 뒤에 줄바꿈 문자 추가


# 정규화까지 한 댓글들 불러와서 vacab_size를 찾아야함
with open("output.txt", "r", encoding="utf-8") as file:
    lines = file.read().splitlines()

df = pd.DataFrame({'comment': lines})

df.columns = ['comment']
normarlized_comments = df['comment'].tolist()


def filter_comments_by_keywords(comments, keywords):
    filtered_comments = []
    for comment in comments:
        for keyword in keywords:
            if keyword in comment:
                filtered_comments.append(comment)
                break  # 이미 한 키워드에 해당하면 다른 키워드 검사하지 않음
    return filtered_comments


def not_in_keywords(comments, keywords):
    no_items = []
    for comment in comments:
        if not all(keyword in comment for keyword in keywords):
            no_items.append(comment)
    return no_items

# 키워드를 포함하는 댓글 필터링


tops_words = ['니트', '셔츠', '상의', '티', '티셔츠', '스웨터', '남방', '목폴라', '폴라']
top_comments = filter_comments_by_keywords(normarlized_comments, tops_words)

bottoms_words = ['스커트', '슬랙스', '슬렉스', '치노',
                 '바지', '조거', '청바지', '치마', '팬츠', '하의']
bottom_comments = filter_comments_by_keywords(
    normarlized_comments, bottoms_words)

shoes_words = ['구두', '운동화', '로퍼', '힐', '스니커즈', '슬리퍼', '샌드', '부츠', '신발']
shoes_comments = filter_comments_by_keywords(
    normarlized_comments, shoes_words)

outer_words = ['패딩', '코드', '자켓', '무스탕', '바람막이', '잠바', '아우터', '오리털', '다운덕', '덕']
outerwear_comments = filter_comments_by_keywords(
    normarlized_comments, outer_words)

accessory_words = ['목걸이', '모자', '안경', '시계', '팔찌', '반지', '목도리', '악세', '악세사리']
accessory_comments = filter_comments_by_keywords(
    normarlized_comments, accessory_words)

all_items = tops_words + bottoms_words + \
    shoes_words + outer_words + accessory_words
not_fashion_items_comments = not_in_keywords(normarlized_comments, all_items)

print(len(top_comments))
print(len(bottom_comments))
print(len(shoes_comments))
print(len(outerwear_comments))
print(len(accessory_comments))
print(len(not_fashion_items_comments))

# # 각 카테고리별 댓글 리스트
# # 예: top_comments = ['댓글1', '댓글2', ...]

# 각 카테고리별 댓글을 데이터프레임으로 변환
df_top = pd.DataFrame({'comment': top_comments, 'label': 0})
df_bottom = pd.DataFrame({'comment': bottom_comments, 'label': 1})
df_outerwear = pd.DataFrame({'comment': outerwear_comments, 'label': 2})
df_shoes = pd.DataFrame({'comment': shoes_comments, 'label': 3})
df_accessory = pd.DataFrame({'comment': accessory_comments, 'label': 4})
df_not_fashion_items = pd.DataFrame(
    {'comment': not_fashion_items_comments, 'label': 5})


# 모든 데이터프레임 결합
df_combined = pd.concat(
    [df_top, df_bottom, df_outerwear, df_shoes, df_accessory, df_not_fashion_items], ignore_index=True)


# 빈도수 낮은 단어들을 찾기 위한 토큰화 과정
mecab = MeCab()
temp = []
for sentence in df_combined['comment']:
    temp_X = mecab.morphs(sentence)
    temp_X = [word for word in temp_X if not word in new_stopwords]  # 불용어 제거
    temp.append(temp_X)


# Tokenizer 정의 및 훈련
tokenizer = Tokenizer()
tokenizer.fit_on_texts(temp)

# 단어 빈도수 계산
threshold = 15
total_cnt = len(tokenizer.word_index)  # 전체 단어 수
rare_cnt = 0  # 빈도수가 기준 이하인 단어 수
rare_freq = 0  # 기준 미만의 단어 빈도수 총합

for key, value in tokenizer.word_counts.items():
    if value < threshold:
        rare_cnt += 1
        rare_freq += value

print('전체 단어 수:', total_cnt)
print('빈도수가 {}번 미만인 희귀 단어 수: {}'.format(threshold, rare_cnt))
print('희귀 단어 비율: {:.2f}%'.format((rare_cnt / total_cnt) * 100))
print('희귀 단어 등장 비율: {:.2f}%'.format(
    (rare_freq / sum(tokenizer.word_counts.values())) * 100))

# 희귀 단어를 제외한 단어 사전 생성
vocab_size = total_cnt - rare_cnt
# 현재 vocap_size는 2950임 하드코딩할려고 할려면 이대로 그대로 쓰면됨

print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@vacab_size는@@@@@@@@@@@@@@@@@@@@@2 : ', vocab_size)


# # 정규화까지 진행된 댓글리스트를 다시 가져와서 0~4까지 분류
temp_2 = []
for sentence in df_combined['comment']:
    temp_X = mecab.morphs(sentence)
    temp_X = [word for word in temp_X if not word in new_stopwords]  # 불용어 제거
    temp_2.append(temp_X)


tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(temp_2)

# 텍스트 시퀀스를 정수 시퀀스로 변환
sequences = tokenizer.texts_to_sequences(temp_2)

# 패딩
max_sequence_length = 150
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)


# 레이블 추출
labels = df_combined['label'].values


# 모델 구축
print('어휘사전 개수:', vocab_size)
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128,
          input_length=max_sequence_length))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(6, activation='softmax'))  # 5개의 출력 클래스

# 모델 컴파일
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])


# 훈련 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.20, random_state=42)


time5 = time.time()

# # 모델 학습
# model.fit(X_train, y_train, batch_size=32, epochs=10,
#           validation_data=(X_test, y_test))


# time6 = time.time()
# print('학습하는데 걸린시간: ', time6 - time5)

# # 모델을 HDF5 파일로 저장
# model.save("classfy_model.h5")


# 저장된 모델 로드
loaded_model = load_model("classfy_model.h5")


mecab = MeCab()
# 댓글들 다시 다시 토큰화 및 패딩까지 진행 한 후 lstm 모델에 넘겨서 전체 댓글을 분류
temp_2 = []
for sentence in df['comment']:
    temp_X = mecab.morphs(sentence)
    temp_X = [word for word in temp_X if not word in new_stopwords]  # 불용어 제거
    temp_2.append(temp_X)


tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(temp_2)

# 텍스트 시퀀스를 정수 시퀀스로 변환
sequences = tokenizer.texts_to_sequences(temp_2)

# 패딩
max_sequence_length = 150
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)


# 댓글에 대한 예측 수행
predictions = loaded_model.predict(padded_sequences)
predicted_labels = np.argmax(predictions, axis=1)

# 예측 결과를 데이터프레임으로 변환
classified_comments_df = pd.DataFrame({
    'comment': df['comment'],
    'label': predicted_labels
})


print(classified_comments_df)


conn = get_db_connection()
# Pandas DataFrame을 PostgreSQL 테이블에 삽입하는 함수


def copy_from_stringio(conn, df, table):
    """
    데이터프레임을 문자열 버퍼로 변환하고 SQL COPY 명령을 사용하여 PostgreSQL 테이블에 삽입
    """
    # 문자열 버퍼 생성
    buffer = StringIO()
    df.to_csv(buffer, index=False, header=False)
    buffer.seek(0)

    # PostgreSQL 삽입을 위한 커서 생성
    cursor = conn.cursor()

    try:
        # COPY 명령 실행
        cursor.copy_from(buffer, table, sep=",", columns=df.columns)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("Data inserted successfully")
    cursor.close()


# 데이터프레임을 PostgreSQL 테이블에 삽입
copy_from_stringio(conn, classified_comments_df, 'classified_comments')


# 파이썬 3.9 설치 텐서플로우 2.11.1 설치, 해야함 박경준 띄어쓰기 정규화 설치하려면.....


'''



챗 gpt 로 현제 프롬프트를 만드니, 현재 딱히 아이템 분류가 안되어도 그냥 gpt 가 알아서 해준다..
현재 gpt가 해주는 거
    1. 주어진 패션 아이템중에서 알아서 뽑아서 모델 스타일링
    2. 혹시라도 번역이 이상하게 된 영어 단어도 알아서 걸러서 들음
    3. 이미지봇에게 줄 프롬프트 작성
    4. 
gpt 없이 할려면.. 일단 패션 아이템 카테고리도 겹치지 않아야 하고...
.. 
댓글 차원에서 분류를 해야 하는데.. 
레이블없이... 비지도 학습으로 ... 댓글을 분류하고 



'''
