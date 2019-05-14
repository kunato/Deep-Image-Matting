import numpy as np
import os
import cv2


def get_trimap(name):
    filename = os.path.join('data/trimap', f'{name}.png')
    alpha = cv2.imread(filename, 0)
    return alpha
