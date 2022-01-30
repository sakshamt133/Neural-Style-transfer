import cv2 as cv
import utilis


img = cv.imread("styl.jpg")
style = utilis.image_process(img)
