import cv2 as cv
import utilis

img = cv.imread("content_mona.jpg")
content = utilis.image_process(img)
