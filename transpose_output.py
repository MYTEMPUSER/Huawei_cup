mat1 = [[8 * j + i for i in range(8)] for j in range(8)]
mat2 = [[8 * i + j for i in range(8)] for j in range(8)]

f_in = open("output.txt", "r")
f_out = open("new_out.txt", "w")
cnt = 0
for line in f_in:
	cnt += 1
	if cnt % 2 == 0:
		print(line)
		data =  list(map(int, line.strip().split(' ')))
		print(data)
		for ind in data:
			for i in range(8):
				for j in range(8):
					if mat2[i][j] == ind:
						f_out.write(str(mat1[i][j]) + " ")
		f_out.write("\n")
	else:
		f_out.write(line)