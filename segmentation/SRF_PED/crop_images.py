# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2018-04-02 19:21:50
# @Last Modified by:   vamshi
# @Last Modified time: 2018-04-20 10:33:57

import os 
import shutil
import sys
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np


data_dir = "./data_dir/"

train_dir = "./train_i/"
train_dir_new = "./train/"
train_label_dir = "./train_labels/"

val_dir = "./val_i/"
val_dir_new = "./val/"
val_label_dir = "./val_labels/"

test_dir = "./test/"
test_dir_new = "./test/"
test_label_dir = "./test_labels/"

new_label_dir = "./label_dir_tf/"

'''for file in os.listdir(data_dir):
	img = cv2.imread(os.path.join(data_dir,file),0)
	img = img[100:612,:]
	cv2.imwrite(data_dir+file,img)'''

for file in sorted(os.listdir(test_dir)):
	print(file)
	main_img = cv2.imread(os.path.join(test_dir,file))
	main_img = main_img[150:662]
	cv2.imwrite(test_dir_new+file[:-4]+".jpg",main_img)
	'''
	img = cv2.imread(os.path.join(train_label_dir,file),1)
	b,g,r = cv2.split(img)
	ret,b = cv2.threshold(b,127,255,cv2.THRESH_BINARY)
	ret,g = cv2.threshold(g,127,255,cv2.THRESH_BINARY)
	ret,r = cv2.threshold(r,127,255,cv2.THRESH_BINARY)
	img = cv2.merge((b,g,r))
	img = img[150:662,:]
	cv2.imwrite(train_label_dir+file[:-4]+".png",img)
	'''

'''
for file in sorted(os.listdir(val_dir)):
	#main_img = cv2.imread(os.path.join(val_dir,file))
	#main_img = main_img[150:662]
	#cv2.imwrite(val_dir_new+file[:-4]+".png",main_img)
	print(file)
	img = cv2.imread(os.path.join(val_label_dir,file[:-3]+"png"),1)
	print(img.shape)
	b,g,r = cv2.split(img)
	ret,b = cv2.threshold(b,127,255,cv2.THRESH_BINARY)
	ret,g = cv2.threshold(g,127,255,cv2.THRESH_BINARY)
	ret,r = cv2.threshold(r,127,255,cv2.THRESH_BINARY)
	img = cv2.merge((b,g,r))
	img = img[150:662,:]
	cv2.imwrite(val_label_dir+file[:-4]+".png",img)

for file in sorted(os.listdir(test_dir)):
	main_img = cv2.imread(os.path.join(test_dir,file))
	main_img = main_img[150:662]
	cv2.imwrite(test_dir_new+file[:-4]+".png",main_img)

	img = cv2.imread(os.path.join(test_label_dir,file),1)
	b,g,r = cv2.split(img)
	ret,b = cv2.threshold(b,127,255,cv2.THRESH_BINARY)
	ret,g = cv2.threshold(g,127,255,cv2.THRESH_BINARY)
	ret,r = cv2.threshold(r,127,255,cv2.THRESH_BINARY)
	img = cv2.merge((b,g,r))
	#img = img[150:662,:]
	cv2.imwrite(test_label_dir+file[:-4]+".png",img)
'''
'''
palette = {(0,   0,   0) : 0 ,
         (255,  0, 0) : 1 ,
         (0,  255,  0) : 2}

def convert_from_color_segmentation(arr_3d):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d        

label_files = os.listdir(label_dir)
for l_f in tqdm(label_files):
    arr = np.array(Image.open(label_dir + l_f))
    arr_2d = convert_from_color_segmentation(arr)
    Image.fromarray(arr_2d).save(new_label_dir + l_f)

print(np.array(Image.open(new_label_dir+os.listdir(new_label_dir)[1])).shape)
'''