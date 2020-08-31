################################
# CAUTION
# ADVANCED MACHINE LEARNING
# DO NOT TOUCH
################################

import torch
from os import walk
import numpy as np
import os
from PIL import Image, ImageDraw 
import math
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.image import imread

device = torch.device("cpu")

# torch.load(map_location=torch.device('cpu'))

part_size = 16

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)

class deep_NNet (nn.Module):    
    def __init__(self):
        super(deep_NNet, self).__init__()
        
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(32, 64, 7, 1, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),    
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            
            View(-1, np.prod(64 * 2 * 4)),
            torch.nn.Linear(64 * 2 * 4, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 2),
            torch.nn.Softmax()
        ).to()
        
    def predict (self, batch):
        return self.net(batch)

PATH = 'model_deep11'

model = deep_NNet()
# model.load_state_dict(torch.load(PATH))
model.eval()

#model = deep_NNet()
#model = torch.load("model_deep6.pt", map_location=device)

#model.eval()
model.double()
model.to(device)

def connect_parts (im1, im2, side):
    im1 = np.array(im1)
    im2 = np.array(im2)

    im1 = np.squeeze(im1)
    im2 = np.squeeze(im2)

    if side == 'tb':
        # print(im1.shape)
        # print(im1.shape)
        
        if im1.shape == (part_size, part_size, 3):
            im1 = im1.transpose(1, 0, 2)
            im2 = im2.transpose(1, 0, 2)
        else:
            im1 = im1.transpose(1, 0)
            im2 = im2.transpose(1, 0)

    res = np.vstack((im1, im2))
    return res

success_cnt = 0

def test_images(image1, image2, side):
    # image = Image.open(path)
    # pix = image.load()
    # print("OK")
    # ind = 0

    X = connect_parts(image1, image2, side)
    mem_X = X
    if np.array(image1).shape == (part_size, part_size, 3):
        mem_X = mem_X.reshape(-1, 3)
        data = mem_X.tolist()
        data = [(i[0], i[1], i[2]) for i in data]
        image = Image.new("RGB", (part_size, part_size * 2))
    else:
        mem_X = mem_X.reshape(-1)
        data = mem_X.tolist()
        data = [(i) for i in data]
        image = Image.new("RGB", (part_size, part_size * 2))
    
    global success_cnt
    image.putdata(data)
    image = image.transpose(Image.ROTATE_270)
    new_name = "fragments/tmp_{}.png".format(success_cnt)
    image.save(new_name)

    X_data = []
    X_data.append(imread(new_name))
    X = np.array(X_data)
    X = X.transpose([0,3,1,2]).astype('float32')
    X = torch.from_numpy(X).type(torch.DoubleTensor)
    # res = model.predict(X.cuda()).detach().cpu().tolist()[0][1]
    res = model.predict(X).detach().cpu().tolist()[0][1]
    # if res < 0.5:
    os.remove(new_name)
    # else:
        # success_cnt += 1
    # print('res', res)
    # if res > 0.5:
    #     print(res)
    return res > 0.5
    # return (lr_connection_matrix[i][j] >= 0.5)


##########################################
##########################################
##########################################


from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from scipy.stats import mstats

import heapq
import math
import time

image_size = 512

class HuaweiImageCollection:
    def __init__(self, fragment_size, shuffled_images_path, answers_path=None, original_images_path=None):
        self.fragment_size = fragment_size
        (_, _, file_names) = next(os.walk(shuffled_images_path))
        file_names = sorted(file_names)
        # print(file_names)
        
        name_to_answer = {name : None for name in file_names}
        if answers_path:
            with open(answers_path, 'r') as answers_file:
                lines = answers_file.readlines()
                for i in range(len(lines) // 2):
                    filename = lines[2 * i].strip()
                    answer = list(map(int, lines[2 * i + 1].split()))
                    name_to_answer[filename] = answer

        self.huawei_images = []
        for name in file_names:
            shuffled_image_path = os.path.join(shuffled_images_path, name)
            original_image_path = None
            if original_images_path:
                original_image_path = os.path.join(original_images_path, name)

            self.huawei_images.append(HuaweiImage(self.fragment_size, shuffled_image_path, name_to_answer[name], original_image_path))
            # print(name)

class HuaweiImage:
    def __init__(self, fragment_size, shuffled_image_path, answer=None, original_image_path=None):
        self.fragment_size = fragment_size
        self.fragments_cnt_sqrt = image_size // self.fragment_size
        self.fragments_cnt = self.fragments_cnt_sqrt ** 2
        
        self.image_name = os.path.split(shuffled_image_path)[-1]
        self.shuffled_image_path = shuffled_image_path
        self.answer = answer
        self.original_image_path = original_image_path

    def load(self):
        self.shuffled_image = Image.open(self.shuffled_image_path)
        if self.original_image_path:
            self.original_image = Image.open(self.original_image_path)
        else:
            self.original_image = None

        # self.shuffled_fragments = None
        # self.shuffled_fragments_2d = None

        data = np.asarray(self.shuffled_image.getdata())
        # print(data.shape)

        if isinstance(data[0], np.integer):
            # self.pixel_dim = 1

            data = [int(x * 256 / 65536) for x in data.tolist()]
            data = [(x, x, x) for x in data]
        # else:
        self.pixel_dim = 3

        print('pixel_dim', self.pixel_dim)

        fragment_side_cnt = image_size // self.fragment_size
        # tuple_or_number = lambda x: [x, x, x] if isinstance(x, np.integer) else x

        data = np.asarray([[[[
            # tuple_or_number(data[(i * self.fragment_size + ii) * image_size + j * self.fragment_size + jj])
            data[(i * self.fragment_size + ii) * image_size + j * self.fragment_size + jj]
            for jj in range(self.fragment_size)]
            for ii in range(self.fragment_size)]
            for j in range(fragment_side_cnt)]
            for i in range(fragment_side_cnt)])

        # print(data.shape)
        data = data.reshape((fragment_side_cnt * fragment_side_cnt, self.fragment_size, self.fragment_size, self.pixel_dim))
        # print(data.shape)

        # print(data.shape)
        # frag = data[0].reshape(self.fragment_size * self.fragment_size, self.pixel_dim)
        # print(frag.shape)
        # print(frag)

        # tmp = Image.fromarray(frag.astype('uint8'))
        # tmp = Image.new("RGB", (self.fragment_size, self.fragment_size))
        # tmp.putdata([tuple(li) for li in frag.tolist()])
        # tmp.show()

        self.shuffled_fragments = data
        # print(self.shuffled_fragments.shape)

class HuaweiImagePrinter:
    def compose_image(self, huawei_image, fragments_matrix):
        data = np.zeros((image_size, image_size, huawei_image.pixel_dim), dtype=int)
        if fragments_matrix is None:
            print('fail')
            tmp = Image.new("RGB", (image_size, image_size))
            return tmp

        f_cnt = huawei_image.fragments_cnt_sqrt
        for i in range(f_cnt):
            for j in range(f_cnt):
                # print(huawei_image.shuffled_fragments.shape)
                # print(fragments_matrix.shape)
                frag = huawei_image.shuffled_fragments[fragments_matrix[i][j]]
                for ii in range(huawei_image.fragment_size):
                    for jj in range(huawei_image.fragment_size):
                        data[i * huawei_image.fragment_size + ii][j * huawei_image.fragment_size + jj] = frag[ii][jj]

        # print(data.shape)
        data = data.reshape((image_size ** 2, huawei_image.pixel_dim))
        # print(data.shape)
        # print(data)

        # some troubles with drawing a grayscale image
        tmp = Image.new("RGB", (image_size, image_size))
        if huawei_image.pixel_dim == 1:
            data = [int(x[0] * 256 / 65536) for x in data.tolist()]
            data = [(x, x, x) for x in data]
            tmp.putdata(data)
        else:
            tmp.putdata([tuple(li) for li in data.tolist()])
        return tmp

    def __init__(self, huawei_image, fragments_matrix=None, output_file_name=None):

        self.compose_image(huawei_image, fragments_matrix).save(output_file_name)
        return 

        huawei_image.load()
        figures_cnt = 2 + (huawei_image.original_image is not None)

        # figure = plt.figure(figsize=(1, figures_cnt))
        figure = plt.figure(figsize=(20, 10))
        
        figure.add_subplot(1, figures_cnt, 1)
        plt.title('shuffled')
        plt.imshow(huawei_image.shuffled_image)

        figure.add_subplot(1, figures_cnt, 2)
        plt.title('FEFU14')
        plt.imshow(self.compose_image(huawei_image, fragments_matrix))

        font = {
            'family': 'serif',
            'color':  'red',
            'weight': 'normal',
            'size': 10,
        }

        frsz = huawei_image.fragment_size
        h = image_size // frsz
        for i in range(h):
            for j in range(h):
                frag_idx = fragments_matrix[i][j]
                y = i * frsz + (frsz // 2)
                x = j * frsz
                s = '{:.0f}'.format(huawei_image.frag_to_homo_coef[frag_idx])
                plt.text(x, y, s, fontdict=font)

        if huawei_image.original_image:
            figure.add_subplot(1, figures_cnt, 3)
            plt.title('original')
            plt.imshow(huawei_image.original_image)
            
        figure.canvas.manager.full_screen_toggle()
        plt.savefig('solved_images_2/' + str(huawei_image.fragment_size) + '_' + huawei_image.image_name + '_' + time.strftime("%d-%m-%y %H:%M:%S", time.gmtime()) + '.png')
        # plt.show()

field_cnt = 0

class HuaweiFieldPrinter:
    def __init__(self, huawei_image, fragments_matrix):
        # huawei_image.load()
        imsz = 2 * image_size

        data = np.zeros((imsz, imsz, huawei_image.pixel_dim), dtype=int)
        if fragments_matrix is None:
            print('fail')
            tmp = Image.new("RGB", (imsz, imsz))
            return tmp

        sz = fragments_matrix.shape[0]
        frsz = huawei_image.fragment_size
        for i in range(sz):
            for j in range(sz):
                if fragments_matrix[i][j] > 0:
                    frag = huawei_image.shuffled_fragments[fragments_matrix[i][j]]
                    for ii in range(frsz):
                        for jj in range(frsz):
                            data[i * frsz + ii][j * frsz + jj] = frag[ii][jj]
                
        # print(data.shape)
        data = data.reshape((imsz ** 2, huawei_image.pixel_dim))
        # print(data.shape)
        # print(data)

        # some troubles with drawing a grayscale image
        tmp = Image.new("RGB", (imsz, imsz))
        if huawei_image.pixel_dim == 1:
            data = [int(x[0] * 256 / 65536) for x in data.tolist()]
            data = [(x, x, x) for x in data]
            tmp.putdata(data)
        else:
            tmp.putdata([tuple(li) for li in data.tolist()])
        
        global field_cnt
        tmp.save('fields/{}.png'.format(field_cnt))
        field_cnt += 1

class HuaweiEasySolver:
    def __init__(self, fragment_size, name='EasySolver'):
        self.fragment_size = fragment_size
        self.name = name
        self.INF = 1e12

    def _calc_distance_matrix(self, huawei_image, output_file_path=None):
        
        def pixel_delta(a, b):
            if huawei_image.pixel_dim == 1:
                return math.fabs(a[0] - b[0])
            return math.fabs(a[0] - b[0]) + math.fabs(a[1] - b[1]) + math.fabs(a[2] - b[2])
            # return a[0] + b[0]# + a[1] + b[1]
            # return np.sum(a)
            # return np.sum(np.absolute(a - b))

        # vector_diff_arr = [-1, 0, 1]
        # vector_coef_arr = [0.5, 1, 0.5]

        def calc_vector_distance(a, b):
            res = 0

            # optimize for first and last pixels
            res += min(pixel_delta(a[0], b[0]), pixel_delta(a[0], b[1]))
            res += min(pixel_delta(a[-1], b[-1]), pixel_delta(a[-1], b[-2]))

            # res += pixel_delta(a[0], b[0]) * vector_coef_arr[1] + pixel_delta(a[0], b[1]) * vector_coef_arr[2]
            # res += pixel_delta(a[-1], b[-1]) * vector_coef_arr[1] + pixel_delta(a[-1], b[-2]) * vector_coef_arr[0]

            for i in range(1, len(a) - 1):
                # res += sum([pixel_delta(a[i], b[i + vector_diff_arr[k]]) * vector_coef_arr[k] for k in range(3)])
                res += min([pixel_delta(a[i], b[i + d]) for d in [-1, 0, 1]])
            return res

        # dist[i][j][k] = distance between fragments 'i' and 'j' (less is better)
        # if 'k' == 0 : 'i' fragment is left, 'j' fragment is right
        # if 'k' == 1 : 'i' fragment is up, 'j' fragment is down

        dist = np.ones((huawei_image.fragments_cnt, huawei_image.fragments_cnt, 2))
        pixel_layer_thickness = 2

        for i in range(huawei_image.fragments_cnt):
            print('metrics', i, 'from', huawei_image.fragments_cnt)
            if i == 10:
                break;
            for j in range(huawei_image.fragments_cnt):
                if i == j:
                    dist[i][i][0] = dist[i][i][1] = self.INF
                    continue

                fragi = huawei_image.shuffled_fragments[i]
                fragj = huawei_image.shuffled_fragments[j]


                # ----------------------
                # basic stuff
                # ---------------------
                # LAST column of 'i' fragment and FIRST column of 'j' fragment
                coli = fragi[:, -1]#.reshape(huawei_image.pixel_dim * huawei_image.fragment_size)
                colj = fragj[:, 0]#.reshape(huawei_image.pixel_dim * huawei_image.fragment_size)
                
                # LAST row of 'i' fragment and FIRST row of 'j' fragment
                rowi = fragi[-1, :]#.reshape(huawei_image.pixel_dim * huawei_image.fragment_size)
                rowj = fragj[0, :]#.reshape(huawei_image.pixel_dim * huawei_image.fragment_size)


                dist[i][j][0] = calc_vector_distance(coli, colj)
                dist[i][j][1] = calc_vector_distance(rowi, rowj)                

        # quit(0)
        if output_file_path:
            with open(output_file_path, 'w') as f:
                c = huawei_image.fragments_cnt
                f.write("\n".join(["\n".join([" ".join(["{:.1f}".format(dist[i][j][k]) for j in range(c)]) for i in range(c)]) for k in [0, 1]]))
        return dist

    def solve_image(self, huawei_image, output_file, dist_precalc=None, output_image_file=None):
        print('solve_image')
        # print('image.load')
        # huawei_image.load()
        print('calc_distance_matrix')
        if dist_precalc is None:
            print('here')
            dist = self._calc_distance_matrix(huawei_image)       
        else:
            dist = dist_precalc
        print('---calc_distance_matrix DONE')

        # --------------------------
        # ALGORITHM
        # 0) result will be stored in matrix 'result', which has shape (2 * sz, 2 * sz)
        #      so it can store matrix of shape (sz, sz), when we start building component
        #      from the center
        # 1) find two nearest fragments, put them in the center of 'result' matrix and
        #      put them in the set 'used'
        # 2) while !used.contains_all_fragments:
        #      search for 'non-used' fragment that is the nearest to the component of 'used'
        #      fragments and connect it to our component

        EMPTY_CELL = -1
        CANDIDATE_CELL = -2
        sz = huawei_image.fragments_cnt_sqrt
        result = np.zeros(dtype=int, shape=(2 * sz, 2 * sz))
        result.fill(EMPTY_CELL)

        USE_MACHINE_LEARNING = True

        # homogeneous_class - less is better
        # homogeneous_classes_cnt = 4
        # homogeneous_threshold = [50, 200, 600]
        # homogeneous_threshold_to_class = [3, 0, 1, 2]

        # homogeneous_classes_cnt = 3
        # homogeneous_threshold = [75, 700]
        # homogeneous_threshold = [85, 750]
        # homogeneous_threshold = [40, 250]
        # homogeneous_threshold = [50, 600]
        # homogeneous_threshold_to_class = [2, 0, 1]

        homogeneous_classes_cnt = 2
        homogeneous_threshold_to_class = [1, 0]
        homogeneous_threshold = [THRESHOLD_PARAMETER]

        min_row, max_row, min_col, max_col = sz, sz, sz, sz
        di = [-1, 1, 0, 0]
        dj = [0, 0, -1, 1]
        
        candidate_cells = [set() for i in range(homogeneous_classes_cnt)]
        cell_to_homo_class = [[-1 for j in range(2 * sz)] for i in range(2 * sz)]
        bad_candidate_cells = []

        remains = set([i for i in range(huawei_image.fragments_cnt)])
        remain_cnt = huawei_image.fragments_cnt

        # oprimization, do not compute distances to the component every time
        # store (distance, version) so we can use heap
        cell_distance = np.zeros((huawei_image.fragments_cnt, 2 * sz, 2 * sz))
        cell_distance_version = np.zeros((huawei_image.fragments_cnt, 2 * sz, 2 * sz), dtype=int)
        cell_distance.fill(-1)
        
        fragment_is_used = np.zeros(dtype=bool, shape=huawei_image.fragments_cnt)
        cell_is_used = np.zeros(dtype=bool, shape=(2 * sz, 2 * sz))

        # list of cells, for which we need to update the distances
        cells_to_update = []
        # heap = []
        cell_heaps = [[[] for j in range(2 * sz)] for i in range(2 * sz)]

        def get_homogeneous_coef(frag_idx):
            sz = huawei_image.fragment_size
            frag = huawei_image.shuffled_fragments[frag_idx]

            f = lambda a: [(np.min([pixel[j] for pixel in a]), np.max([pixel[j] for pixel in a])) for j in range(3)]
            s = lambda x: np.sum([x[i][1] - x[i][0] for i in range(3)])

            # res = np.min([s(f(frag[0,:])), s(f(frag[-1,:])), s(f(frag[:,0])), s(f(frag[:,-1]))])
            tmp = sorted([s(f(frag[0,:])), s(f(frag[-1,:])), s(f(frag[:,0])), s(f(frag[:,-1]))])
            res = np.mean(tmp[1:-1])
            return res

        def get_homogeneous_class(frag_idx):
            return homogeneous_coef_to_class(get_homogeneous_coef(frag_idx))

        def homogeneous_coef_to_class(coef):
            for i in range(homogeneous_classes_cnt - 1):
                if coef < homogeneous_threshold[i]:
                    return homogeneous_threshold_to_class[i]
            return homogeneous_threshold_to_class[-1]

        def remove_candidate_cell(row, col):
            for i in range(homogeneous_classes_cnt):
                candidate_cells[i].discard((row, col))

        def add_candidate_cell(row, col, homo_class):
            candidate_cells[homo_class].add((row, col))
            cell_to_homo_class[row][col] = homo_class

        def update_candidate_cell(row, col, homo_class):
            if homo_class < cell_to_homo_class[row][col]:
                candidate_cells[cell_to_homo_class[row][col]].discard((row, col))
                add_candidate_cell(row, col, homo_class)

        def put_fragment(fragment_idx, row, col):
            nonlocal min_row, max_row, min_col, max_col, remain_cnt

            min_row = min(min_row, row)
            max_row = max(max_row, row)
            min_col = min(min_col, col)
            max_col = max(max_col, col)

            assert(result[row][col] < 0)
            if result[row][col] == CANDIDATE_CELL:
                remove_candidate_cell(row, col)
            result[row][col] = fragment_idx
            remains.discard(fragment_idx)
            remain_cnt -= 1

            fragment_is_used[fragment_idx] = True
            cell_is_used[row][col] = True

            for z in range(4):
                i = row + di[z]
                j = col + dj[z]
                if bad_cell(i, j):
                    continue

                if result[i][j] == EMPTY_CELL:
                    result[i][j] = CANDIDATE_CELL
                    add_candidate_cell(i, j, frag_to_homo_class[fragment_idx])

                if result[i][j] == CANDIDATE_CELL:
                    cell_distance[fragment_idx][i][j] = -1
                    cells_to_update.append((i, j))
                    update_candidate_cell(i, j, frag_to_homo_class[fragment_idx])

        def bad_cell(row, col):
            return \
                row - min_row >= sz or \
                max_row - row >= sz or \
                col - min_col >= sz or \
                max_col - col >= sz  

        def get_cell_distance(frag_idx, i, j):
            # if cell_distance[frag_idx][i][j][0] >= 0:
            #     return cell_distance[frag_idx[i][j]]

            cur_distances = []
            # top cell
            if not bad_cell(i - 1, j) and result[i - 1][j] >= 0:
                cur_distances.append(dist[result[i - 1][j]][frag_idx][1])
            # bottom cell
            if not bad_cell(i + 1, j) and result[i + 1][j] >= 0:
                cur_distances.append(dist[frag_idx][result[i + 1][j]][1])
            # left cell
            if not bad_cell(i, j - 1) and result[i][j - 1] >= 0:
                cur_distances.append(dist[result[i][j - 1]][frag_idx][0])
            # right cell
            if not bad_cell(i, j + 1) and result[i][j + 1] >= 0:
                cur_distances.append(dist[frag_idx][result[i][j + 1]][0])

            assert(cur_distances)
            cur_distance = np.mean(cur_distances)
            return cur_distance

        def update_cells():
            nonlocal cells_to_update, remains, cell_heaps

            for i, j in cells_to_update:
                if not bad_cell(i, j):
                    for frag_idx in remains:
                        cell_distance[frag_idx][i][j] = get_cell_distance(frag_idx, i, j)
                        cell_distance_version[frag_idx][i][j] += 1

                        # !!! TODO: structure instead of this 5-element tuple???
                        mega_tuple = (cell_distance[frag_idx][i][j], cell_distance_version[frag_idx][i][j], frag_idx, i, j)
                        heapq.heappush(cell_heaps[i][j], mega_tuple)
                        # heapq.heappush(heap, mega_tuple)

            cells_to_update = []

        def bad_heap_element(cell_dist, dist_version, frag_idx, i, j):
            return fragment_is_used[frag_idx] or cell_is_used[i][j] or \
                bad_cell(i, j) or \
                dist_version != cell_distance_version[frag_idx][i][j]

        def remove_bad_heap_elements(he):
            while bad_heap_element(*he[0]):
                heapq.heappop(he)

        def pick_top_k_from_heap(he, k):
            # print('k={}'.format(k))
            # print('heap', heap)
            # print()
            res = []
            for i in range(k):
                remove_bad_heap_elements(he)
                res.append(he[0])
                heapq.heappop(he)
            for e in res:
                heapq.heappush(he, e)
            return res

        def ud_fragments_near(a, b):
            return test_images(huawei_image.shuffled_fragments[a], huawei_image.shuffled_fragments[b], 'lr')

        def lr_fragments_near(a, b):
            return test_images(huawei_image.shuffled_fragments[a], huawei_image.shuffled_fragments[b], 'tb')

        def find_nearest(row, col):
            k = min(25, remain_cnt)
            top = pick_top_k_from_heap(cell_heaps[row][col], k)
            # top = list(remains)

            if USE_MACHINE_LEARNING:
                for e in top:
                    frag_idx = e[2]
                    # frag_idx = e

                    # neighbours = 0
                    # good = 0

                    # check top-to-bottom relations
                    i = row - 1
                    j = col
                    if not bad_cell(i, j) and result[i][j] != -1:        
                        # neighbours += 1
                        if ud_fragments_near(result[i][j], frag_idx):
                            # good += 1
                            return frag_idx
                  
                    # check bottom-to-top relations
                    i = row + 1
                    j = col
                    if not bad_cell(i, j) and result[i][j] != -1:
                        # neighbours += 1
                        if ud_fragments_near(frag_idx, result[i][j]):
                            # good += 1
                            return frag_idx

                    # check left-to-right relations
                    i = row
                    j = col - 1
                    if not bad_cell(i, j) and result[i][j] != -1:
                        # neighbours += 1
                        if lr_fragments_near(result[i][j], frag_idx):
                            # good += 1
                            return frag_idx

                    # check left-to-right relations
                    i = row
                    j = col + 1
                    if not bad_cell(i, j) and result[i][j] != -1:
                        # neighbours += 1
                        if lr_fragments_near(frag_idx, result[i][j]):
                            # good += 1
                            return frag_idx

                    # if good >= (neighbours + 1) // 2:
                        # return frag_idx

            # found nothing interesting => return nearest element according to the shit-metric
            # print('---using shit-metric')
            return top[0][2]

        def remove_bad_candidate_cells():
            for i, j in bad_candidate_cells:
                assert(result[i][j] < 0)
                remove_candidate_cell(i, j)
                result[i][j] = EMPTY_CELL

        # pick a cell (i, j) with maximum cell_heaps[i][j] among all candidate cells
        def pick_optimal_cell():
            for homo_class in range(homogeneous_classes_cnt):
                if candidate_cells[homo_class]:
                    res = (self.INF, -1, -1)
                    for row, col in candidate_cells[homo_class]:
                        if bad_cell(row, col):
                            bad_candidate_cells.append((row, col))
                        else:
                            res = min(res, (pick_top_k_from_heap(cell_heaps[row][col], 1)[0][0], row, col))
                    
                    if res[-1] != -1:
                        return res

        def pick_the_first_optimal_fragment():
            res = (homogeneous_classes_cnt, -1, -1)
            for i in range(huawei_image.fragments_cnt):
                res = min(res, (frag_to_homo_class[i], np.min(dist[i]), i))
            return res[2]

        frag_to_homo_class = [get_homogeneous_class(i) for i in range(huawei_image.fragments_cnt)]
        frag_to_homo_coef = [get_homogeneous_coef(i) for i in range(huawei_image.fragments_cnt)]
        huawei_image.frag_to_homo_coef = frag_to_homo_coef

        # print(frag_to_homo_class[:16])
        # for i in range(16):
        #     print(i, ')', frag_to_homo_class[i], get_homogeneous_coef(i))
        # quit(0)

        # print('finding first_pair')

        first_fragment = pick_the_first_optimal_fragment()
        # first_pair = np.unravel_index(dist.argmin(), dist.shape)
        put_fragment(first_fragment, sz, sz)
        update_cells()
        # print('first_fragment', first_fragment)

        # left-to-right
        # if first_pair[2] == 0:
        put_fragment(find_nearest(sz, sz + 1), sz, sz + 1)
        # top-to-bottom
        # else:
            # put_fragment(find_nearest(sz + 1, sz), sz + 1, sz)

        # print('sz', sz)
        print('start the main loop')
        main_loop_iter_idx = 0
        while remains:
            print('STEP {} from {}'.format(main_loop_iter_idx, huawei_image.fragments_cnt))
            main_loop_iter_idx += 1

            bad_cells = []
            min_distance = self.INF
            min_frag_idx = -1
            min_cell = (-1, -1)

            update_cells()
            remove_bad_candidate_cells()

            #
            # pick the nearest cell (according to the shit-metric)
            # then try to use ultracool advanced machine learning instrtument
            # 

            # !!!!!!!!!!
            # winner = pick_top_k_from_heap(heap, 1)[0]
            min_cell = pick_optimal_cell()
            # print('winner_cell distance =', min_cell[0])
            min_frag_idx = find_nearest(min_cell[1], min_cell[2])

            assert(min_frag_idx != -1)
            assert(not fragment_is_used[min_frag_idx])
            put_fragment(min_frag_idx, min_cell[1], min_cell[2])
            
            # HuaweiFieldPrinter(huawei_image, result)

        print('loop done')

        matrix = result[min_row : max_row + 1, min_col : max_col + 1]
        # print(matrix)
        # print(huawei_image.answer)
        # print(result)

        printable_matrix = matrix.reshape(matrix.shape[0] ** 2)
        if output_file:
            with open(output_file, 'a') as f:
                f.write(huawei_image.image_name + "\n")
                f.write(" ".join(list(map(str, printable_matrix.tolist()))) + "\n")

        if output_image_file:
            HuaweiImagePrinter(huawei_image, matrix, output_image_file)


solve_size = 16
image_index = 223
THRESHOLD_PARAMETER = 80

test_collection = HuaweiImageCollection(
    solve_size,
    'shuffled-images-data/data_test2_blank/{}'.format(solve_size)
    )

solver = HuaweiEasySolver(solve_size)

output_file_name = 'test_output_{}.txt'.format(solve_size)
open(output_file_name, 'w')

# solver.solve_image(test_collection.huawei_images[image_index], output_file_name)
# quit(0)

def read_matrix(file):
    sz = image_size // solve_size
    with open(file, 'r') as f:
        f.readline()
        a = np.asarray(list(map(int, f.readline().split())))
        a = a.reshape((sz, sz))
    return a

# HuaweiImagePrinter(test_collection.huawei_images[image_index], read_matrix("/home/danilov/Desktop/huawei/calg/cmake-build-debug/answer.txt"))
# solver.solve_image(test_collection.huawei_images[image_index], output_file_name)
# quit(0)

# for i in range(len(test_collection.huawei_images)):
#     print("\nIMAGE # {} \n".format(i))
#     solver.solve_image(test_collection.huawei_images[i], output_file_name)

def load_distance_matrix(huawei_image, filename):
    dist = np.zeros((huawei_image.fragments_cnt, huawei_image.fragments_cnt, 2))
    
    with open(filename, 'r') as f:
        for i in range(huawei_image.fragments_cnt):
            cur = f.readline().split()
            for j in range(huawei_image.fragments_cnt):
                dist[i][j][0] = cur[j]

        for i in range(huawei_image.fragments_cnt):
            cur = f.readline().split()
            for j in range(huawei_image.fragments_cnt):
                dist[i][j][1] = cur[j]

    return dist

def bruteforce_theshold(huawei_image, l, r, step):
    global THRESHOLD_PARAMETER
    huawei_image.load()
    dist = load_distance_matrix(huawei_image, 'distances_new/' + huawei_image.image_name + '.txt')
    for THRESHOLD_PARAMETER in range(l, r + 1, step):
        solver.solve_image(huawei_image, 'mega_folder/{}.txt'.format(huawei_image.image_name), dist, 'mega_folder/{}_{}.png'.format(huawei_image.image_name, THRESHOLD_PARAMETER))

for i in range(len(test_collection.huawei_images)):
    print("\nIMAGE # {} \n".format(i))
    bruteforce_theshold(test_collection.huawei_images[i], 50, 70, 10)