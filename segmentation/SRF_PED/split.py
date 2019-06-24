# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2018-05-03 15:34:06
# @Last Modified by:   vamshi
# @Last Modified time: 2018-05-04 10:34:08

import cv2
import os,sys
import numpy as np
from random import shuffle
import random

data_dir = "./data"
data_labels_dir = "./data_labels"

train_dir = "./train/"
train_labels_dir = "./train_labels/"

val_dir = "./val/"
val_labels_dir = "./val_labels/"

test_dir = "./test/"
test_labels_dir = "./test_labels/"

data = os.listdir(data_dir)
data_files = random.sample(data,len(data))

n_data = len(data_files)

n_train = int(n_data*0.8)+1
n_val = int(n_train*0.1)
n_test = int(n_data*0.2)

train = data_files[0:n_train]
test  = data_files[n_train:n_train+n_test]
train = random.sample(train, len(train))
#val  = train[0:n_val]
#train = train[n_val:]
val = test

for file in train:
	img = os.path.join(data_dir,file)
	img_label = os.path.join(data_labels_dir,file[:-4]+".png")

	os.system("cp "+img+" "+train_dir)
	os.system("cp "+img_label+" "+train_labels_dir)

'''
for file in test:
	img = os.path.join(data_dir,file)
	img_label = os.path.join(data_labels_dir,file[:-4]+".png")

	os.system("cp "+img+" "+test_dir)
	os.system("cp "+img_label+" "+test_labels_dir)
'''

for file in val:
	img = os.path.join(data_dir,file)
	img_label = os.path.join(data_labels_dir,file[:-4]+".png")

	os.system("cp "+img+" "+val_dir)
	os.system("cp "+img_label+" "+val_labels_dir)


#print(train) 
print len(train),len(test),len(val)
