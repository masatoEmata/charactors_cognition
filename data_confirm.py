# 実装時の挙動確認用

from importlib.resources import path
from statistics import mode
import numpy as np
from PIL import Image
import pickle
from google.colab.patches import cv2_imshow
import cv2

def pickle_show(save_file, index):
    with open(save_file, mode="rb") as f:
        hoge = pickle.load(f)
        label = hoge[index][0]
        im_array = hoge[index][1]

        print(label)
        cv2_imshow(im_array)


demo_paths = [
'/content/sample_data/ETL8B/emata_netsu.PNG',
'/content/sample_data/ETL8B/emata_imouto.png',
'/content/sample_data/ETL8B/emata_da.png',
'/content/sample_data/ETL8B/emata_yoko.png'
]

class UseModel:
    def __init__(self, model, im_size=32, im_color=1) -> None:
        self.model = model
        self.im_size = im_size
        self.im_color = im_color

    def reshape(self, path):
        img = cv2.imread(path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img_gray, (self.im_size, self.im_size))
        cv2_imshow(img)
        img = img.astype('float').reshape(self.im_size, self.im_size, self.im_color) / 255
        return img

    def predict(self, paths=demo_paths):
        imgs = [self.reshape(path) for path in paths]
        np_imgs = np.array(imgs)

        predictions = self.model.predict(np_imgs)
        [prd.argmax() for prd in predictions]