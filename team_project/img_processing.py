import os
import cv2
import numpy as np

'''
크기 조정 (Resizing):

딥러닝 모델은 고정된 크기의 이미지를 입력으로 사용하므로 모든 이미지를 동일한 크기로 조정합니다.
색상 정규화 (Color normalization):

이미지 픽셀 값의 범위를 [0, 1] 또는 [-1, 1]로 조정하여 모델의 학습을 안정화하고, 수렴 속도를 빠르게 합니다.
데이터 증강 (Data Augmentation):

오버피팅을 방지하고 모델의 일반화 능력을 향상시키기 위해 원본 이미지에 여러 변형을 추가합니다. 회전, 확대/축소, 가로/세로 이동, 반전 등의 방법이 있습니다.
밝기, 명암 조정:

일부 이미지가 너무 어둡거나 밝을 수 있으므로, 이미지의 밝기나 명암을 조절하여 모델에 더 좋은 데이터를 제공할 수 있습니다.
노이즈 제거:

이미지에 포함된 노이즈를 제거하기 위해 Gaussian blur 등의 기술을 사용할 수 있습니다.
색상 공간 변환 (Color space transformation):

RGB, HSV, Grayscale 등 다양한 색상 공간으로 이미지를 변환하여 모델의 성능을 향상시킬 수 있는지 확인합니다.

케라스의 ImageDataGenerator 사용하기

'''


def processingFunc(base_dir_path, effect_name, effectFunc):
    base_dir = base_dir_path
    emotions = ['angry', 'happy', 'sad', 'relaxed']

    for emotion in emotions:
        img_folder = os.path.join(base_dir, emotion)

        print(img_folder)
        image_names = os.listdir(img_folder)

        print(image_names)
        for img in image_names:
            one_img_path = os.path.join(base_dir, emotion, img)
            # unchanged 붙이는 거 중요
            img_obj = cv2.imread(one_img_path, cv2.IMREAD_UNCHANGED)

            # 여기서 이미지에 처리할 것들 하고
            # 지금 현재는 리사이징
            # output_img = cv2.resize(img_obj, (350, 350))
            effected_img = effectFunc(img_obj)

            # 출력 폴더 생성
            if not os.path.exists("output_folder"):
                os.makedirs("output_folder")

            # 해당 이모션 폴더 만들고
            emotion_folder = os.path.join("output_folder", emotion)
            if not os.path.exists(emotion_folder):
                os.makedirs(emotion_folder)

            # output_img = cv2.
            # 파일 이름 만들어줘야함.
            # img는 원본 이미지 파일의 파일이름임.
            # 확장자를 제외한 파일 이름만 가져옵니다.
            without_ext_name = os.path.splitext(img)[0]
            file_ext = os.path.splitext(img)[1]
            file_name = without_ext_name + '_' + effect_name + file_ext
            print(file_name)

            # cv2.imwrite(os.path.join(emotion_folder, file_name), output_img)
            if not cv2.imwrite(os.path.join(emotion_folder, file_name), effected_img):
                print("Failed to save image!")








def filterFunc(img):
    return cv2.Canny(img, 30, 70)

# 원본 사진을 그냥 output폴더로 옮겨주는 함수
def pass_origin(img):
    return img



# 기본 이미지 전처리들
'''
- **크기 조절**: 이미지의 해상도를 변경합니다.
- **회전과 이동**: 이미지를 회전하거나 위치를 이동시킵니다.
- **색상 조정**: 밝기, 대비, 채도, 명도 등을 조정합니다.
- **히스토그램 평활화**: 이미지의 히스토그램을 평활화하여 대비를 개선합니다.
'''

# 크기 조절
# resize img
def resize_func(img):
    # type what to do to original images here
    return cv2.resize(img, (384, 384))


# 회전
# rotate img
def random_rotateFunc(img):
    rows, cols = img.shape[:2]
    random_angle = np.random.randint(1, 36) * 10
    # 첫번째: 회전 중심점, 두번째: 회전 각도, 
    # 세번째: 화면 확대/축소 비율(1은 원본, 2는 2배 0.5는 절반크기)    
    M = cv2.getRotationMatrix2D((cols/2, rows/2), random_angle, 1)
    return cv2.warpAffine(img, M, (cols, rows))


# 흑백으로 바꾸기
def to_grayscale(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 흑백으로 바꿔도 채널은 계속 3채널로 유지되는 경우가 있다.
    # 다음 코드로 채널을 1채널로 바꾼다.
    if len(gray_img.shape) == 3:
        gray_img = gray_img[:, :, 0]
    
    return gray_img
# 흑백 이미지 평탄화 시키기
def to_flat(img):
    # 이미지가 그레이스케일이 아닌 경우 그레이스케일로 변환
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img

    # 히스토그램 평활화 수행
    return cv2.equalizeHist(gray_img)




processingFunc("output_folder", "to_flat", to_flat)




# img ='./tDKYiQEAoyo9bKAeZtij2dnxR8A564497_to_gray.jpg'

# import cv2
# image = cv2.imread(img)

# print(image.shape)
# print(image.dtype)


