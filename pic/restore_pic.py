


import cv2
import os

dir_path = os.path.join(os.getcwd(),'..','input_data_2')
pic_names = os.listdir(dir_path)
train_path = os.path.join(os.getcwd(),'train')
test_path = os.path.join(os.getcwd(),'test')
# print(pic_names)
for name in pic_names:
	fpath = os.path.join(dir_path,name)
	img = cv2.imread(fpath)
	print(img)
	[sort,num] = name.split('_')
	if int(num.split('.')[0]) % 10 != 0:

		if not os.path.exists(os.path.join(train_path, sort)):
			os.mkdir(os.path.join(train_path, sort))
		p = os.path.join(train_path, sort, num)
		cv2.imencode('.jpg', img)[1].tofile(p)
	else:
		p = os.path.join(test_path, sort + '_' + num)
		cv2.imencode('.jpg', img)[1].tofile(p)

