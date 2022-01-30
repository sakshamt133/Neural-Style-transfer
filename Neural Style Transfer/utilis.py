from torchvision import transforms
import cv2 as cv
import numpy as np

in_channels = 3
img_size = 400
epochs = 40
lr = 0.001
layers = [1, 3, 5, 9, 14, 17]
alpha = 0.5
beta = 10

transform = transforms.Compose([
    transforms.ToTensor()
])


def image_process(img):
    img = cv.resize(img, (img_size, img_size))
    img = np.asarray(img)
    content = transform(img)
    content = content.unsqueeze(0)
    return content
