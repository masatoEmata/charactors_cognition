# Based on: https://qiita.com/kcrt/items/a7f0582a91d6599d164d


#!/usr/bin/env python3

import struct
import numpy as np
import os
from PIL import Image


ETL_TYPE = 'ETL8G'
IMG_DIR = f"/content/sample_data/{ETL_TYPE}/img"
SOURCE_DIR = f"/content/sample_data/{ETL_TYPE}/source"
# SOURCE_DIR = f"dataset/{ETL_TYPE}/source"
RECORD_SIZE = 8199
FILE_CNT = 33

for cnt in range(2, FILE_CNT):
    if cnt + 1 < 10:
        cnt_str = f'0{cnt+1}'
    else:
        cnt_str = str(cnt+1)
    file_name = f'{ETL_TYPE}_{cnt_str}'
    file_path = f'{SOURCE_DIR}/{file_name}'
    i = 0
    print("Reading {}".format(file_path))
    with open(file_path, 'rb') as f:
        while True:
            s = f.read(RECORD_SIZE)
            if s is None or len(s) < RECORD_SIZE:
                break
            r = struct.unpack(">HH8sIBBBBHHHHBB30x8128s11x", s)
            img = Image.frombytes('F', (128, 127), r[14], 'bit', (4, 0))
            img = img.convert('L')
            img = img.point(lambda x: 255 - (x << 4))
            i = i + 1
            dirname = b'\x1b$B' + r[1].to_bytes(2, 'big') + b'\x1b(B'
            dirname = dirname.decode("iso-2022-jp")
            try:
                os.makedirs(IMG_DIR)
            except:
                pass
            imagefile = f"{IMG_DIR}/{file_name}_{i:0>6}.png"
            print(imagefile)
            img.save(imagefile)
