import os
from shutil import copyfile

### list based on https://github.com/senguptaumd/Background-Matting/blob/master/Data_adobe/train_data_list.txt
train_list = open('train_data_list.txt').read().splitlines()

### training
src_fg_path = '/export/ccvl12b/qihang/MGMatting/data/Combined_Dataset/Training_set/fg/'
src_alpha_path = '/export/ccvl12b/qihang/MGMatting/data/Combined_Dataset/Training_set/alpha/'

dst_fg_path = '/export/ccvl12b/qihang/MGMatting/data/Combined_Dataset_Solid/Training_set/fg/'
dst_alpha_path = '/export/ccvl12b/qihang/MGMatting/data/Combined_Dataset_Solid/Training_set/alpha/'

if not os.path.exists(dst_fg_path):
    os.makedirs(dst_fg_path)
if not os.path.exists(dst_alpha_path):
    os.makedirs(dst_alpha_path)

for f in train_list:
    copyfile(src_fg_path + f, dst_fg_path + f)
    copyfile(src_alpha_path + f, dst_alpha_path + f)


