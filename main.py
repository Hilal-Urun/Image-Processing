# Image enhancement with pillow
# Color , Sharpness, Brightness, Contrast with factor 1.5
from PIL import Image, ImageEnhance
# import cv2
import numpy as np

cam = cv2.VideoCapture(0)


def image_enhancer(frame):
    frame = Image.fromarray(frame)
    colored_image = ImageEnhance.Color(frame).enhance(1.5)
    sharpned_image = ImageEnhance.Sharpness(colored_image).enhance(1.5)
    bright_image = ImageEnhance.Brightness(sharpned_image).enhance(1.5)
    contrast_image = ImageEnhance.Contrast(bright_image).enhance(1.5)
    return np.asarray(contrast_image)


while cam.isOpened():
     _, frame = cam.read()
     frame = image_enhancer(frame)
     cv2.imshow("frame", frame)
     if cv2.waitKey(1) & 0xFF == ord('q'):
         break

 # Release capture
cam.release()
cv2.destroyAllWindows()
