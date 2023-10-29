from tensorflow.keras.models import Sequential
# 다른 tensorflow.keras 관련 모듈도 이와 같이 변경

from tensorflow.keras.layers import Embedding, Dense, LSTM
from mecab import MeCab
import re
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from konlpy.tag import Mecab
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_file = urllib.request.urlopen(
    "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt")
test_file = urllib.request.urlopen(
    "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt")

train_data = pd.read_table(train_file)
test_data = pd.read_table(test_file)

print(train_data[:10])

# 중복 제거
train_data.drop_duplicates(subset=['document'], inplace=True)
# 널 값 제거
train_data = train_data.dropna(how='any')
# 한글, 공백을 제외하고 모두 제거
train_data['document'] = train_data['document'].str.replace(
    "[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
train_data[:10]
# 비어있는거는 널로 전환해서 지워버리기
train_data['document'].replace('', np.nan, inplace=True)
train_data = train_data.dropna(how='any')
print('학습 데이터 개수:', len(train_data))

# 검증 데이타도 똑같이 정리
test_data.drop_duplicates(subset=['document'], inplace=True)
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
test_data['document'].replace('', np.nan, inplace=True)
test_data = test_data.dropna(how='any')
test_data[:10]
print('검증 데이터 개수:', len(test_data))


# 토큰화 및 불용어 제거
tagger = MeCab()

stopwords = ['다', '고', '하', '을', '보', '것', '음', '나', '게', '지', '있', '의', '이',
             '가', '은', '들', '는', '과', '하다', '한다', '에', '에서', '로', '으로', '와', '도', '한', '를']
X_train = []
for sentence in train_data['document']:
    X_train.append([word for word in tagger.morphs(
        sentence) if not word in stopwords])

X_test = []
for sentence in test_data['document']:
    X_test.append([word for word in tagger.morphs(
        sentence) if not word in stopwords])

# 토큰나이저로 인덱스 붙이기
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

print(tokenizer.word_index)


# 낮은 빈도수 제거
threshold = 3
words_cnt = len(tokenizer.word_index)
rare_cnt = 0

words_freq = 0
rare_freq = 0

for key, value in tokenizer.word_counts.items():
    words_freq = words_freq + value

    if value < threshold:
        rare_cnt += 1
        rare_freq = rare_freq + value


print("전체 단어 수: ", words_cnt)
print("빈도가 {}미만인 단어 수: {}".format(threshold-1, rare_cnt), words_cnt)
print('희귀 단어 비율: {}'.format((rare_cnt / words_cnt) * 100))
print('회귀 단어 등장 빈도 비율 {}'.format((rare_freq / words_freq) * 100))


vocab_size = words_cnt - rare_cnt
print('실제 사용하는 단어수:', vocab_size)


tokenizer = Tokenizer(vocab_size, oov_token='OOV')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])


drop_train = [index for index, sentence in enumerate(
    X_train) if len(sentence) < 1]

X_train = [sentence for index, sentence in enumerate(
    X_train) if index not in drop_train]
y_train = [label for index, label in enumerate(
    y_train) if index not in drop_train]


print(len(X_train))
print(len(y_train))


max_len = 60
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)


print('리뷰최대길이:', max(len(i) for i in X_train))
print('리뷰평균길이:', sum(map(len, X_train))/len(X_train))

model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@어디서 문제인거지??????????')
# history = model.fit(X_train, y_train, epochs=10,
#                     batch_size=60, validation_split=0.2)
history = model.fit(X_train, y_train, epochs=10,
                    batch_size=60, validation_split=0.2, verbose=0)

# 예측


def sent_pred(new_sentence):
    new_token = [word for word in tagger.morphs(
        new_sentence) if not word in stopwords]
    new_sequences = tokenizer.texts_to_sequences([new_token])
    new_pad = pad_sequences(new_sequences, maxlen=max_len)

    score = float(model.predict(new_pad, verbose=0))

    if score > 0.5:
        print("{} => 긍정({:.2f}%)".format(new_sentence, score * 100))
    else:
        print("{} => 부정({:.2f}%)".format(new_sentence, (1 - score) * 100))


sent_pred("뭐야 이 평점들은...  나쁘진 않지만 10점 짜리는 더더욱 아니잖아 ")
