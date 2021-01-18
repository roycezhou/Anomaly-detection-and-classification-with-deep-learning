import numpy as np
from PIL import Image
import cv2

class Equalizer(object):
    def __init__(self):
        pass
    def __call__(self, image):
        image_array = np.asarray(image)
        image_array.setflags(write=1)
        image_array[:,:,0] = cv2.equalizeHist(image_array[:,:,0])
        image_array[:,:,1] = cv2.equalizeHist(image_array[:,:,1])
        image_array[:,:,2] = cv2.equalizeHist(image_array[:,:,2])
        return Image.fromarray(np.uint8(image_array))