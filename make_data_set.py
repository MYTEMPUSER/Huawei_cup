from PIL import Image, ImageDraw 
import math
import cv2
import numpy as np
from os import walk
import pandas as pd
import random

image_size = 512
INF = 100000000000000000
part_size = 32

lr_used = {}
tb_used = {}
data_set = pd.DataFrame(columns=["image", "res"])

def split_image (image):
	parts = [[[[0 for j in range(part_size)] for i in range(part_size)] for part_x in range(image_size // part_size)] for part_y in range(image_size // part_size)]
	for i in range(int(image_size / part_size)):
		for j in range(int(image_size / part_size)):
			for x in range(part_size):
				for y in range(part_size):
					pix = image[i * part_size + x, j * part_size + y]
					parts[i][j][x][y] = pix #0.299 * pix[0]  + 0.587 * pix[1] + 0.114 * pix[2]
	return parts

def connect_parts (im1, im2, ind, side, RES):
	im1 = np.array(im1)
	im2 = np.array(im2)
	#print(res.shape)
	global data_set
	if side == 'tb':
		if im1.shape == (part_size, part_size, 3):
			im1 = im1.transpose(1, 0, 2)
			im2 = im2.transpose(1, 0, 2)
			res = np.hstack((im1, im2))
		else:
			im1 = im1.transpose(1, 0)
			im2 = im2.transpose(1, 0)
			res = np.hstack((im1, im2))

	res = np.vstack((im1, im2))

	if im1.shape == (part_size, part_size, 3):
		res = res.reshape(-1, 3)
		data = res.tolist()
		data = [(i[0], i[1], i[2]) for i in data]
		image = Image.new("RGB", (part_size, part_size * 2))
	else:
		res = res.reshape(-1)
		data = res.tolist()
		data = [(i) for i in data]
		image = Image.new("RGB", (part_size, part_size * 2))

	image.putdata(data)
	image = image.transpose(Image.ROTATE_270)
	image.save("data_set\\" + str(ind) + ".png")
	data_set = data_set.append({"image": str(ind) + ".png", "res": RES}, ignore_index=True)

ind = 0
def make_data (path, name):
	image = Image.open(path)
	images = [image]
	global ind
	for image in images:
		pixs = image.load()
		parts = split_image(pixs)

		for i in range(len(parts)):
			for j in range(len(parts)):
				if i != len(parts) - 1:
					ind += 1
					connect_parts(parts[i][j], parts[i + 1][j], ind, "lr", 1)
				if j != len(parts) - 1:
					ind += 1
					connect_parts(parts[i][j], parts[i][j + 1], ind, "tb", 1)
				if i != len(parts) - 1 and j != len(parts) - 1 and random.randint(0, 1) == 0:
					ind += 1
					connect_parts(parts[i][j], parts[i + 1][j + 1], ind, "lr", 0)
					ind += 1
					connect_parts(parts[i][j], parts[i + 1][j + 1], ind, "tb", 0)
				else:
					a_x = random.randint(0, len(parts) - 1)
					a_y = random.randint(0, len(parts) - 1)
					b_x = random.randint(0, len(parts) - 1)
					b_y = random.randint(0, len(parts) - 1)
					if (b_x - a_x == 1 and b_y - a_y == 0) or (b_x - a_x == 0 and b_y - a_y == 1):
						pass
					else:
						ind += 1
						connect_parts(parts[a_x][a_y], parts[b_x][b_y], ind, "lr", 0)
						ind += 1
						connect_parts(parts[a_x][a_y], parts[b_x][b_y], ind, "tb", 0)



for (dirpath, dirnames, filenames) in walk("C:\\Users\\dvfu\\Desktop\\make_data_set\\data"):
	for file in filenames:
		print(file)
		make_data(dirpath + '\\' + file, file)

data_set.to_csv("data.csv")