# 実装時の挙動確認用
import numpy as np
import cv2
from google.colab.patches import cv2_imshow


demo_paths = [
    '/content/emata_a.png'
]

class UseModel:
    def __init__(self, model, im_size=32, im_color=1) -> None:
        self.model = model
        self.im_size = im_size
        self.im_color = im_color

    def format_img(self, path):
        # print('path', path)
        print('path', path)
        img = cv2.imread(path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img_gray, (self.im_size, self.im_size))
        cv2_imshow(img)
        img = img.astype('float').reshape(self.im_size, self.im_size, self.im_color) / 255
        return img

    def predict(self, paths=demo_paths):
        print(paths)
        imgs = [self.format_img(path) for path in paths if type(path) != 'str' or path is not None]
        # cv2_imshow(imgs[0])
        np_imgs = np.array(imgs)
        predictions = self.model.predict(np_imgs)
        return [prd.argmax() for prd in predictions]

model_handler = UseModel(model)
predictions = model_handler.predict()
print(predictions)