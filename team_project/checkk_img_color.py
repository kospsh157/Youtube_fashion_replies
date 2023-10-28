import cv2

# imread는 기본적으로 bgr형태로 이미지를 읽는다. 따라서 이렇게 해서 나온 이미지가 정상적으로 보인다면,
# 그 이미지는 bgr형태의 이미지인것이다.
img = cv2.imread(
    './DogEmotion/angry/0aNyXBrmNA7XdefwHvgO2n1rnpqQAp885_to_hsv.jpg')

cv2.imshow('img_window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 정상적으로 보이므로 형태 받은 데이터셋은 bgr 타입이다
