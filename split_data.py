import os
import random
from config.config import WEBSITES_DATASET_PATH
from shutil import copyfile

BASE_FOLDER = '/workspace/data_local/websites2'
if not os.path.isdir(BASE_FOLDER):
	os.mkdir(BASE_FOLDER)

TRAIN_FOLDER = '/workspace/data_local/websites2/train'
if not os.path.isdir(TRAIN_FOLDER):
	os.mkdir(TRAIN_FOLDER)

VAL_FOLDER = '/workspace/data_local/websites2/val'
if not os.path.isdir(VAL_FOLDER):
	os.mkdir(VAL_FOLDER)

for folder in os.listdir(WEBSITES_DATASET_PATH):
	print(folder)
	source_folder = os.path.join(WEBSITES_DATASET_PATH, folder)
	
	dest_val_folder = os.path.join(VAL_FOLDER, folder)
	if not os.path.isdir(dest_val_folder):
		os.mkdir(dest_val_folder)

	dest_train_folder = os.path.join(TRAIN_FOLDER, folder)
	if not os.path.isdir(dest_train_folder):
		os.mkdir(dest_train_folder)

	filelist = os.listdir(source_folder)
	random.shuffle(filelist)
	for idx in range(200):
		src = os.path.join(source_folder, filelist[idx])
		dst = os.path.join(dest_val_folder, filelist[idx])
		copyfile(src, dst)
	for idx in range(200,1200):
		src = os.path.join(source_folder, filelist[idx])
		dst = os.path.join(dest_train_folder, filelist[idx])
		copyfile(src, dst)