from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout

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


model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu',
          input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

time_1 = time.time()
model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator)
time_2 = time.time()

print(f'학습시간: {time_2 - time_1}초')

# If you run this code more than once, make sure to comment out the section that splits the training and test data before executing.
