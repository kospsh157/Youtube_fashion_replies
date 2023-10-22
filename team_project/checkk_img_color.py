import cv2 

img = cv2.imread('./DogEmotion/angry/0aNyXBrmNA7XdefwHvgO2n1rnpqQAp885.jpg')

cv2.imshow('img_window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()