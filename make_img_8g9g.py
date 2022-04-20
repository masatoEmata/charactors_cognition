# Based on: https://qiita.com/kcrt/items/a7f0582a91d6599d164d


#!/usr/bin/env python3

import struct
import os
from PIL import Image


ETL_TYPE = 'ETL8G'
IM_DIR = f"/content/sample_data/{ETL_TYPE}/img"
# IM_DIR = f"./dataset/{ETL_TYPE}/img"
SOURCE_DIR = f"/content/sample_data/{ETL_TYPE}/source"
# SOURCE_DIR = f"dataset/{ETL_TYPE}/source"
RECORD_SIZE = 8199
FILE_CNT = 33


def read_record_ETL(f):
    s = f.read(RECORD_SIZE)
    if s is None or len(s) < RECORD_SIZE:
        return None
    r = struct.unpack(">HH8sIBBBBHHHHBB30x8128s11x", s)
    img = Image.frombytes('F', (128, 127), r[14], 'bit', (4, 0))
    img = img.convert('L')
    img = img.point(lambda x: 255 - (x << 4))
    dirname = b'\x1b$B' + r[1].to_bytes(2, 'big') + b'\x1b(B'
    dirname = dirname.decode("iso-2022-jp")
    return (img, dirname)

def save_image_file(img, dirname):
    try:
        os.makedirs(f"{IM_DIR}/{dirname}")
    except:
        pass
    imagefile = f"{IM_DIR}/{dirname}/{file_name}_{i:0>6}.png"
    img.save(imagefile)

for i, cnt in enumerate(range(FILE_CNT)):
    img_class_i = cnt + 1
    if img_class_i < 10:
        cnt_str = f'0{img_class_i}'
    else:
        cnt_str = str(img_class_i)
    file_name = f'{ETL_TYPE}_{cnt_str}'
    file_path = f'{SOURCE_DIR}/{file_name}'
    with open(file_path, 'rb') as f:
        while True:
            record = read_record_ETL(f)
            if record:
                img, dirname = record
                save_image_file(img, dirname)
            else:
                break
