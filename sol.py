from PIL import Image, ImageDraw 
import math
import cv2

image_size = 512
INF = 100000000000000000
part_size = 32

dict_val = {}

def delta(pix1, pix2):
	if isinstance(pix1, tuple): 
		return math.fabs(pix1[0] - pix2[0]) + \
		   math.fabs(pix1[1] - pix2[1]) + \
		   math.fabs(pix1[2] - pix2[2])
	else:
		return math.fabs(pix1 - pix2)

def calc_mertic (image1, image2, ind1, ind2, side):
	sum = 0
	if (ind1, ind2, side) in dict_val:
		return dict_val[(ind1, ind2, side)]
	if side == 'lr':
		for i in range(part_size):
			min_delta = INF
			for d in [-1, 0, 1]:
				if i + d > -1 and i + d < part_size:
					min_delta = min(delta(image1[part_size - 1][i], image2[0][i + d]), min_delta)
			sum += min_delta
	else:
		for i in range(part_size):
			min_delta = INF
			for d in [-1, 0, 1]:
				if i + d > -1 and i + d < part_size:
					min_delta = min(delta(image1[i][part_size - 1], image2[i + d][0]), min_delta)
			sum += min_delta
	dict_val[(ind1, ind2, side)] = sum
	return sum

def split_image (part_size, image):
	parts = [[[0 for j in range(part_size)] for i in range(part_size)] for part in range(image_size // part_size * image_size // part_size)]
	origi = [[[0 for j in range(part_size)] for i in range(part_size)] for part in range(image_size // part_size * image_size // part_size)]
	for i in range(int(image_size / part_size)):
		for j in range(int(image_size / part_size)):
			for x in range(part_size):
				for y in range(part_size):
					pix = image[i * part_size + x, j * part_size + y]
					origi[i * (image_size // part_size) + j][x][y] = pix
					parts[i * (image_size // part_size) + j][x][y] = pix #0.299 * pix[0]  + 0.587 * pix[1] + 0.114 * pix[2]
	return parts, origi

f = open("output.txt", "w")

def solve(path, name):
	res_matrix = []
	f.write(name + '\n')
	image = Image.open(path)
	pix = image.load()
	parts, orig = split_image(part_size, pix)
	image_1_index = -1
	img = [[0 for j in range(image_size)] for i in range(image_size)]
	best = INF
	cnt = 0
	for image_start in range(len(parts)):
		cnt += 1
		print(cnt)
		matrix = [[0 for i in range(image_size // part_size)] for j in range(image_size // part_size)]
		matrix[0][0] = image_start + 1
		used = {}
		used[image_start] = True
		sum_var = 0
		while len(used) != len(parts):
			candidate = [INF, 0, 0, 0, 0]
			for im in range(len(parts)):
				if im not in used:
					image = parts[im]
					for i in range(len(matrix)):
						for j in range(len(matrix)):
							neibour = 0
							result = 0
							if matrix[i][j] != 0 or i - 1 == -1 or matrix[i - 1][j] == 0:
								pass
							else:
								neibour += 1
								result += calc_mertic(parts[matrix[i - 1][j] - 1], image, matrix[i - 1][j], im + 1, "lr")

							if matrix[i][j] != 0 or j - 1 == -1 or matrix[i][j - 1] == 0:
								pass
							else:
								neibour += 1
								result += calc_mertic(parts[matrix[i][j - 1] - 1], image, matrix[i][j - 1], im + 1, "tb")
							
							if matrix[i][j] != 0 or i + 1 == image_size // part_size or matrix[i + 1][j] == 0:
								pass
							else:
								neibour += 1
								result += calc_mertic(parts[im], parts[matrix[i + 1][j] - 1], im + 1, matrix[i + 1][j], "lr")

							if matrix[i][j] != 0 or j + 1 == image_size // part_size or matrix[i][j + 1] == 0:
								pass
							else:
								neibour += 1
								result += calc_mertic(parts[im], parts[matrix[i][j + 1] - 1], im + 1, matrix[i][j + 1], "tb")

							if neibour == 0:
								result = INF
							else:
								result /= neibour
	
							if result < candidate[0]:
								candidate = [result, im, i, j, neibour]
			matrix[candidate[2]][candidate[3]] = candidate[1] + 1
			used[candidate[1]] = True

		for i in range(len(matrix)):
			for j in range(len(matrix)):
				if i != 0:
					sum_var += calc_mertic(parts[matrix[i - 1][j] - 1], parts[matrix[i][j] - 1], matrix[i - 1][j], matrix[i][j], "lr")
				if j != 0:
					sum_var += calc_mertic(parts[matrix[i][j - 1] - 1], parts[matrix[i][j] - 1], matrix[i][j - 1], matrix[i][j], "tb")

		print(image_start, ':', sum_var)
		if sum_var < best:
			res_matrix = matrix
			best = sum_var
			for j in range(image_size):
				for i in range(image_size):
					img[i][j] = orig[matrix[i // part_size][j // part_size] - 1][i % part_size][j % part_size]

	data = []
	for j in range(image_size):
		for i in range(image_size):	
			data.append(img[i][j])

	for j in range(len(res_matrix)):
		for i in range(len(res_matrix)):
			f.write(str(res_matrix[i][j] - 1) + " ")
	f.write("\n")

	image = Image.new("RGB", (image_size, image_size))
	image.putdata(data)
	image.save(name + "res.png")

from os import walk

for (dirpath, dirnames, filenames) in walk("C:\\Users\\Igor\\Desktop\\Huawei\\data_test1_blank\\32"):
	for file in filenames:
		dict_val = {}
		print(dirpath + '\\' + file)
		solve(dirpath + '\\' + file, file)


