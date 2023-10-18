import datetime as dt
import time
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import numpy as np


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
        shutil.move(src, dst)

    # 검증 데이터 폴더 생성 및 파일 이동
    val_emotion_dir = os.path.join(val_dir, emotion)
    if not os.path.exists(val_emotion_dir):
        os.makedirs(val_emotion_dir)
    for fname in val_files:
        src = os.path.join(emotion_dir, fname)
        dst = os.path.join(val_emotion_dir, fname)
        shutil.move(src, dst)


# 데이터 경로 설정
base_dir = 'output_folder'
train_data_dir = os.path.join(base_dir, 'train')
validation_data_dir = os.path.join(base_dir, 'validation')
img_width, img_height = 200, 200
batch_size = 32


# 이미지 데이터 전처리

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    # color_mode='grayscale',  # 흑백 이미지로 로드
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    # color_mode='grayscale',  # 흑백 이미지로 로드
    batch_size=batch_size,
    class_mode='categorical')


# 모델 구축
model = Sequential()


# 첫 번째 Convolutional 레이어
model.add(Conv2D(16, (3, 3), activation='relu',
          input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 두 번째 Convolutional 레이어
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully Connected 레이어
model.add(Flatten())
model.add(Dense(32, activation='relu'))

# Output 레이어
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# 모델 학습
epochs = 15
time2 = time.time()
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator)
time1 = time.time()

print(f'Learning time: {time1 - time2}')


time2 = time.time()
# Evaluate fucntion
# 모델 성능 평가
loss, accuracy = model.evaluate(validation_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
time1 = time.time()
print(time1 - time2)
