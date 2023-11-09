import os
import cv2
import numpy as np


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


# 모서리감지?
def filterFunc(img):
    return cv2.Canny(img, 30, 70)


# 원본 사진을 그냥 output폴더로 옮겨주는 함수
def pass_origin(img):
    return img


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
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# 흑백 이미지 평탄화 시키기
def to_flat(img):
    grayscale_img = to_grayscale(img)
    return cv2.equalizeHist(grayscale_img)


# 모폴로지 연산
# 열기
def opening(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


# 닫기
def closing(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


# BGR -> HSV
def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# 미디언 블러
def median_blur(img):
    return cv2.medianBlur(img, 7)


# 기하학적 변환
'''
5. **기하학적 변환**:
    - **어파인 변환**: 이미지를 이동, 회전, 스케일링합니다.
    - **원근 변환**: 이미지의 원근 효과를 조절합니다.
4. **색상 및 채널 처리**:
    - **색상 공간 변환**: RGB에서 Grayscale, HSV, LAB 등 다른 색상 공간으로 변환합니다.
    - **채널 분할 및 결합**: 이미지의 개별 채널을 분리하거나 결합합니다.

'''

# 첫번째 인자: 원본이미지 있는 폴더 경로
# 두번째 인자: 새롭게 처리되어 나올 이미지의 이름을 입력
# 예를들어 "closing" 이라고 적으면 처리된 이미지 파일 이름은 "원본파일명_closing.jpg" 이렇게 나옴
# 세번째 인자: 실질적으로 전처리를 하는 함수
processingFunc("DogEmotion", "pass_origin",  pass_origin)


# delete to_gray images but not to_flat images in this folder
# find "$directory_path" -type f -name "*to_gray*" ! -name "*to_flat*" -exec rm {} \;
