#coding:utf-8
"""
生成的jpg大小在0.95到1.0之间
"""
import os
from PIL import Image

jpg_min_size = 0.95
jpg_max_size = 1.0
jpg_init_quality = 15
jpg_adjust_step = 3

tif_list_path = './test/data/tif/002.tif'

jpg_list_path = './002.jpg'

img = Image.open(tif_list_path)

img.convert("L").save(jpg_list_path, quality=jpg_init_quality, dpi=(300.0, 300.0))


cur_quality = jpg_init_quality
op_cnt = 0

while (os.path.getsize(jpg_list_path) * 1.0 / 1024 / 1024) < jpg_min_size:  # unit: metabytes
    cur_quality += jpg_adjust_step
    img.convert("L").save(jpg_list_path, quality=cur_quality, dpi=(300.0, 300.0))
    op_cnt += 1
while (os.path.getsize(jpg_list_path) * 1.0 / 1024 / 1024) > jpg_max_size:  # unit: metabytes
    cur_quality -= jpg_adjust_step
    img.convert("L").save(jpg_list_path, quality=cur_quality, dpi=(300.0, 300.0))
    op_cnt += 1

print('tif:{}->jpg:{},调整次数:{},最终质量:{},最终大小:{}MB'.format(tif_list_path, jpg_list_path, op_cnt, cur_quality, os.path.getsize(jpg_list_path)/1024/1024))
