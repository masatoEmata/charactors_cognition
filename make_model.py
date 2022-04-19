# Based on: https://qiita.com/Cyber_Hacnosuke/items/c121cfd1945a3174bc84


import keras
import pickle
from keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Dense, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D, Input
import numpy as np
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

ETL_TYPE = 'ETL8G'
IM_DIR = f"/content/sample_data/{ETL_TYPE}"
IM_SIZE = 32
SAVE_FILE = IM_DIR + f"/{ETL_TYPE}.pickle"
OUT_SIZE = 979 # アーンまでの文字の数
IM_COLOR = 1
IN_SHAPE = (IM_SIZE, IM_SIZE, IM_COLOR)

# 保存した画像データ一覧を読み込む
data = pickle.load(open(SAVE_FILE, "rb"))

# 画像データを0-1の範囲に直す
y = []
x = []
for d in data:
  (num, img) = d
  # cv2_imshow(img)
  # print(num)
  img = img.astype('float').reshape(IM_SIZE, IM_SIZE, IM_COLOR) / 255 # 255が妥当かは要確認
  y.append(keras.utils.np_utils.to_categorical(num, OUT_SIZE))
  x.append(img)
x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)
#モデル構築
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=IN_SHAPE))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(OUT_SIZE))
model.add(Activation('softmax'))

#コンパイル
model.compile(
  loss='categorical_crossentropy',
  optimizer= 'adam',
  metrics=['accuracy'])

hist = model.fit(
  x_train, y_train,
  batch_size=2048, epochs=100,verbose=1,
    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=1)
print("正解率 ", score[1], "loss ", score[0])

plt.plot(hist.history['accuracy'])
plt.title('Accuracy')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

plt.plot(hist.history['loss'])
plt.title('Loss')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

model.save(f'{ETL_TYPE}-model.h5')
model.save_weights(f'{ETL_TYPE}-weights.h5')