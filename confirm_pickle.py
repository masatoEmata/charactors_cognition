import pickle
from google.colab.patches import cv2_imshow

def pickle_show(save_file, index):
    with open(save_file, mode="rb") as f:
        data = pickle.load(f)
        label = data[index][0]
        im_array = data[index][1]

        print(label)
        cv2_imshow(im_array)