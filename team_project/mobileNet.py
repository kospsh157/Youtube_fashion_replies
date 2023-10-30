import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
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
    # color_mode='grayscale',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    # color_mode='grayscale',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


# MobileNetV2 모델 로드
input_shape = (96, 96, 3)
base_model = MobileNetV2(input_shape=input_shape,
                         include_top=False, weights='imagenet')

# 새로운 분류 레이어 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)

# 사전 훈련된 레이어 동결
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습 (사전 훈련된 레이어의 가중치는 동결 상태)
model.fit(train_generator, epochs=30, validation_data=(validation_generator))

# 사전 훈련된 레이어의 동결 해제
for layer in base_model.layers:
    layer.trainable = True

# 미세 조정을 위한 학습률을 낮춘 새로운 옵티마이저 사용
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

# 미세 조정 학습
model.fit(train_generator, epochs=5, validation_data=(validation_generator))
