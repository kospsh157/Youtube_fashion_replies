import os
import cv2


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
# 애초에 train 폴더에서 시작한다는 것 주의 ...
# 여기서 밑에 부분에 resize 함수 말고 딴거 처리할 코드를 넣으면  output_folder 만들어 지고 자동으로 감정 폴더 만들어
# 해당 폴더에 알아서 넣는다.
# 다만 train 폴더에서 실행해야 하고 그 다음 또 validation 폴더 가서 다시한번 실행해줘야 한다.

def processingFunc(base_dir_path, effect_name, effectFunc):
    base_dir = base_dir_path
    emotions = ['angry', 'happy', 'sad', 'relaxed']

    for emotion in emotions:
        img_folder = os.path.join(base_dir, emotion)
        image_names = os.listdir(img_folder)

        for img in image_names:
            one_img_path = os.path.join(base_dir, emotion, img)
            img_obj = cv2.imread(one_img_path)

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
            without_ext_name = os.path.splitext(img)[0]  # 확장자를 제외한 파일 이름만 가져옵니다.
            file_ext = os.path.splitext(img)[1]
            file_name = without_ext_name + '_' + effect_name + file_ext 
            print(file_name)

            # cv2.imwrite(os.path.join(emotion_folder, file_name), output_img)
            if not cv2.imwrite(os.path.join(emotion_folder, file_name), effected_img):
                print("Failed to save image!")
          


def effectFunc(img):
    # type what to do to original images here
    return cv2.resize(img, (176, 176))
    



processingFunc("DogEmotion", "resize", effectFunc)
