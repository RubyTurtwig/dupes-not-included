import numpy as np
import cv2
import pickle

from PIL import Image


def display_image(image:np.array): 
    """ Display image. """

    while True:
        cv2.imshow('Window', image)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__': 

    with open('tests/screen.pkl', 'rb') as f:
        img = pickle.load(f)

    display_image(img)