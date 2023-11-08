

import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 데이터셋 경로 설정
base_dir = './output_folder'
emotions = ['happy', 'sad', 'relaxed', 'angry']


# Train, Validation 데이터셋 분리
train_datagen = ImageDataGenerator(
    validation_split=0.2,  # 20%를 검증 데이터로 사용
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical',
    subset='training'  # 학습 데이터 부분
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical',
    subset='validation'  # 검증 데이터 부분
)


# InceptionResNetV2 모델
base_model = InceptionResNetV2(weights='imagenet', include_top=False)

# 새로운 출력 레이어 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)


# 블럭 8 전까지는 동결
for layer in base_model.layers[:630]:
    layer.trainable = False
# 'block8 이 인덱스 630 부터 시작이므로 여기 부터는 동결 해제
for layer in model.layers[630:]:
    layer.trainable = True

# 컴파일
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])


# 모델 학습
history_1 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=30
)


# base_model의 모든 레이어를 동결 해제
for layer in base_model.layers:
    layer.trainable = True


# 컴파일 (미세조정)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
history_2 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=30)

# 모델 평가
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")


# 모델 학습 곡선
hist_dict = history_1.history
loss = hist_dict['loss']
val_loss = hist_dict['val_loss']
accuracy = hist_dict['accuracy']
val_accuracy = hist_dict['val_accuracy']

plt.plot(loss, 'b', label='training loss')
plt.plot(val_loss, 'r', label='validation loss')
plt.legend()
plt.grid()

plt.figure()
plt.plot(accuracy, 'b', label='training accuracy')
plt.plot(val_accuracy, 'r', label='validation accuracy')
plt.legend()
plt.grid()

plt.show()


hist_dict_2 = history_2.history
loss_2 = hist_dict_2['loss']
val_loss_2 = hist_dict_2['val_loss']
accuracy_2 = hist_dict_2['accuracy']
val_accuracy_2 = hist_dict_2['val_accuracy']

plt.plot(loss_2, 'b', label='training loss')
plt.plot(val_loss_2, 'r', label='validation loss')
plt.legend()
plt.grid()

plt.figure()
plt.plot(accuracy_2, 'b', label='training accuracy')
plt.plot(val_accuracy_2, 'r', label='validation accuracy')
plt.legend()
plt.grid()

plt.show()
