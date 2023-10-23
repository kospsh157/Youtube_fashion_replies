from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model
import time

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input

import os
import shutil
import numpy as np
import time
import datetime as dt

base_dir = 'output_folder'
emotions = ['angry', 'happy', 'relaxed', 'sad']

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

# train, validation 폴더 생성
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# 각각의 감정 폴더로부터 데이터 분할
val_split = 0.2  # 검증 데이터의 비율
for emotion in emotions:
    emotion_dir = os.path.join(base_dir, emotion)
    emotion_files = os.listdir(emotion_dir)
    np.random.shuffle(emotion_files)

    val_count = int(len(emotion_files) * val_split)
    train_files = emotion_files[val_count:]
    val_files = emotion_files[:val_count]

    # 훈련 데이터 폴더 생성 및 파일 이동
    train_emotion_dir = os.path.join(train_dir, emotion)
    if not os.path.exists(train_emotion_dir):
        os.makedirs(train_emotion_dir)
    for fname in train_files:
        src = os.path.join(emotion_dir, fname)
        dst = os.path.join(train_emotion_dir, fname)
        shutil.copy(src, dst)

    # 검증 데이터 폴더 생성 및 파일 이동
    val_emotion_dir = os.path.join(val_dir, emotion)
    if not os.path.exists(val_emotion_dir):
        os.makedirs(val_emotion_dir)
    for fname in val_files:
        src = os.path.join(emotion_dir, fname)
        dst = os.path.join(val_emotion_dir, fname)
        shutil.copy(src, dst)


# 데이터 경로 설정
base_dir = 'output_folder'
train_data_dir = os.path.join(base_dir, 'train')
validation_data_dir = os.path.join(base_dir, 'validation')
img_width, img_height = 224, 224
batch_size = 32


# 이미지 데이터 전처리
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


base_model = DenseNet121(weights=None, include_top=False,
                         input_tensor=Input(shape=(224, 224, 1)))

# 사용자 정의 출력층 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# 첫 번째 부분의 모델을 고정 (사전 학습된 가중치 변경 안 함)
# (위에서 주석처리 되어 있어서 필요시 주석 해제 후 사용하시면 됩니다.)
# for layer in base_model.layers:
#     layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
time1 = time.time()
model.fit(train_generator, epochs=40, validation_data=validation_generator)
time2 = time.time()
print('Learning Time: ', time2 - time1)

# 모델 평가
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation Loss: {loss:.4f}')
print(f'Validation Accuracy: {accuracy:.4f}')
