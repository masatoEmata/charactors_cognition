# Based on: https://qiita.com/Cyber_Hacnosuke/items/c121cfd1945a3174bc84

# 使用するライブラリを読み込む
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import pickle
import os

ETL_TYPE = 'ETL8B'

# 画像集を保存するディレクトリ
outdir = f"./dataset/{ETL_TYPE}/img/"
# outdir = f"/content/sample_data/{ETL_TYPE}/img/"
im_size = 32 # 画像サイズ

save_file = outdir + f"/{ETL_TYPE}.pickle" # 保存ファイル名と保存先
plt.figure(figsize=(9, 17)) # 出力画像を大きくする

# ひらがな画像集のディレクトリから画像を読み込み開始
files = os.listdir(outdir)
files_dir = [f for f in files if os.path.isdir(os.path.join(outdir, f))]

result = []
for i, code in enumerate(files_dir):
    img_dir = outdir + "/" + str(code)
    fs = glob.glob(img_dir + "/*")
    print("dir=",  img_dir)

    # 画像64X63を読み込んでグレイスケールに変換し32X32に整形
    for j, f in enumerate(fs):
        img = cv2.imread(f)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img_gray, (im_size, im_size))
        result.append([i, img])

        # ひらがな画像一覧表示 10行X5列
        if j == 3:
            plt.subplot(11, 5, i + 1)
            plt.axis("off")
            plt.title(str(i))
            plt.imshow(img, cmap='gray')

# ひらがなの「画像とラベル」のデータセットを保存
pickle.dump(result, open(save_file, "wb"))
plt.show()