# Based on: https://qiita.com/Cyber_Hacnosuke/items/c121cfd1945a3174bc84
# ETL8B 用
import struct
from PIL import Image, ImageEnhance
import glob, os

from make_pickle import ETL_TYPE

RECORD_SIZE = 512
ETL_TYPE = 'ETL8B'

# 画像集を保存するディレクトリ
outdir = f"./dataset/{ETL_TYPE}/img/"
# outdir = f"/content/sample_data/{ETL_TYPE}/img/"
if not os.path.exists(outdir): os.mkdir(outdir)

# ETL8Bディレクトリ内部の4個の分割データを読み込む
files = glob.glob(f"./dataset/{ETL_TYPE}/source/ETL*")
# files = glob.glob(f"/content/sample_data/{ETL_TYPE}/source/ETL*")
fc = 0
for fname in files:
  fc = fc + 1
  print(fname) # ETL8Bの分割ファイル名

  # ETL7の分割ファイル名を開く
  f = open(fname, 'rb')
  f.seek(0)
  i = 0
  while True:
    i = i + 1
    # あいうえおのラベルと画像データの組をRECORD_SIZE byteずつ読む
    s = f.read(RECORD_SIZE)
    if not s: break
    try:
        # バイナリデータなのでPythonが理解できるように
        r = struct.unpack('>2H4s504s', s)
        # 画像として取り出す
        iF = Image.frombytes('1', (64, 63), r[3], 'raw')
        iP = iF.convert('L')
        code_jis = r[3]
        dir = outdir + "/" + str(hex(r[1])[-4:])
        if not os.path.exists(dir): os.mkdir(dir)
        # fn = "{0:02x}-{1:02x}{2:04x}.png".format(code_jis, r[0], r[2])
        fn = f'{ETL_TYPE}_{(r[0]-1)%20+1}.png'
        fullpath = dir + "/" + fn
        #if os.path.exists(fullpath): continue
        enhancer = ImageEnhance.Brightness(iP)
        iE = enhancer.enhance(16)
        iE.save(fullpath, 'PNG')
    except Exception as e:
        print(e)
print("ok")