"""Read the numpy and transform it to an image"""
import numpy as np
import cv2

def read_and_save(image, path_to_save):
    """Take a numpy of size lengthxwidex3, converts it to an image and saves it"""
    img_to_save = (image * 255.0).astype(np.uint8)
    cv2.imwrite(path_to_save, img_to_save)