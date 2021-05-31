#!/usr/bin/python
# encoding: utf-8

import random
from torch.utils.data import Dataset
from config import cfg

import sys
import os
import torch
import numpy as np
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageDraw, Image
import pickle, tqdm
import glob
from collections import Counter
from torchvision import transforms
import matplotlib.pyplot as plt


def read_pot(f_path):
    fp = open(f_path, 'rb')
    samples = []
    all_bytes = fp.read()
    i = 0
    while i < len(all_bytes):
        # read head
        sample_size = int.from_bytes(all_bytes[i:i + 2], sys.byteorder, signed=False)  # bytes size of current character
        i += 2
        tag_code = int.from_bytes(all_bytes[i:i + 4][::-1][-2:], sys.byteorder,
                                  signed=False)  # Dword (int) type, GB2132 or GBK
        tag_char = all_bytes[i:i + 4][::-1][-2:-1].decode('gbk') if tag_code < 256 else all_bytes[i:i + 4][::-1][
                                                                                        -2:].decode('gbk')
        i += 4
        stroke_number = int.from_bytes(all_bytes[i:i + 2], sys.byteorder, signed=False)  # unsigned short type
        i += 2
        #  read stroke coordinate
        coordinates_size = sample_size - 8
        coordinates = []
        stroke = []
        for _ in range(coordinates_size):
            x = int.from_bytes(all_bytes[i:i + 2], sys.byteorder, signed=True)
            i += 2
            y = int.from_bytes(all_bytes[i:i + 2], sys.byteorder, signed=True)
            i += 2
            if (x, y) == (-1, 0):
                coordinates.append(stroke)
                stroke = []
            elif (x, y) == (-1, -1):
                break
            else:
                stroke.extend([x, y])
        assert len(coordinates) == stroke_number, "stroke length should be equal to stroke_number"
        samples.append(
            {'tag': tag_code, 'tag_char': tag_char, 'stroke_number': stroke_number, 'coordinates': coordinates})
    fp.close()
    return samples


def read_gnt(gnt_path):
    samples = []
    fp = open(gnt_path, 'rb')
    all_bytes = fp.read()
    i = 0
    while i < len(all_bytes):
        sample_size = int.from_bytes(all_bytes[i:i + 4], sys.byteorder, signed=False)  # bytes size of current character
        i += 4
        tag_code = int.from_bytes(all_bytes[i:i + 2][-2:], sys.byteorder,
                                  signed=False)  # Dword (int) type, GB2132 or GBK
        tag_char = all_bytes[i:i + 2][::-1][-2:-1].decode('gbk') if tag_code < 256 else all_bytes[i:i + 2].decode('gbk')
        # print(tag_char)
        i += 2
        width = int.from_bytes(all_bytes[i:i + 2], sys.byteorder, signed=False)
        i += 2
        height = int.from_bytes(all_bytes[i:i + 2], sys.byteorder, signed=False)
        i += 2
        bitmap = np.frombuffer(all_bytes[i: i + width * height], dtype=np.uint8)
        bitmap = bitmap.reshape(height, width)
        # cv2.imshow('bitmap', bitmap)
        # cv2.waitKey(0)
        i += width * height
        samples.append({
            'sample_size': sample_size, 'tag_code': tag_code, 'tag_char': tag_char, 'width': width, 'height': height,
            'bitmap': bitmap
        })
    fp.close()
    return samples


def rm_list_repeat_items(input_list):
    repeat_dict = dict(Counter(input_list))
    return [key for key, value in repeat_dict.items() if value == 1]


def intersection_list(list_a, list_b):
    return list(set(list_a).intersection(set(list_b)))


def alignment_gnt_pot(input_gnt, input_pot):
    gnt_tag_list = [temp_sample['tag_code'] for temp_sample in input_gnt]
    pot_tag_list = [temp_sample['tag'] for temp_sample in input_pot]
    gnt_tag_list_rm = rm_list_repeat_items(gnt_tag_list)
    pot_tag_list_rm = rm_list_repeat_items(pot_tag_list)
    intersection_tag_list = intersection_list(gnt_tag_list_rm, pot_tag_list_rm)
    output_gnt = [input_gnt[gnt_tag_list.index(i_tag)] for i_tag in intersection_tag_list]
    output_pot = [input_pot[pot_tag_list.index(i_tag)] for i_tag in intersection_tag_list]
    return output_gnt, output_pot


def coords_render(coordinates, width, height, thickness, board=5):
    canvas_w = width  # 256
    canvas_h = height  # 256
    # 预留 5% 的边框，使得边界的笔画得以完整显示
    board_w = board  # min(1, int(canvas_w * 0.05))
    board_h = board  # min(1, int(canvas_h * 0.05))
    # preprocess canvas size
    p_canvas_w = canvas_w - 2 * board_w
    p_canvas_h = canvas_h - 2 * board_h
    # find original character size to fit with canvas
    min_x = 635535
    min_y = 635535
    max_x = -1
    max_y = -1
    for stroke in coordinates:
        for (x, y) in np.array(stroke).reshape((-1, 2)):
            min_x = min(x, min_x)
            max_x = max(x, max_x)
            min_y = min(y, min_y)
            max_y = max(y, max_y)
    original_size = max(max_x - min_x, max_y - min_y)
    # 拉伸轨迹尺度到(p_canvas_w, p_canvas_h)之内，使其能塞入画布中，并预留边框
    canvas = Image.new(mode='L', size=(canvas_w, canvas_h), color=255)
    draw = ImageDraw.Draw(canvas)
    new_coords = []
    for i, stroke in enumerate(coordinates):
        xys = np.array(stroke, dtype=np.float)
        xys[::2] = (xys[::2] - min_x) / original_size * p_canvas_w + board_w  # + board_w为预留边框
        # xys[1::2] = xys[1,3,5,...]
        xys[1::2] = (xys[1::2] - min_y) / original_size * p_canvas_h + board_h
        xys = np.round(xys)  # 坐标点只能是整数
        draw.line(xys.tolist(), fill=0, width=thickness)
        new_coords.append(xys)
    w_ratio = 1. / original_size * p_canvas_w
    h_ratio = 1. / original_size * p_canvas_h
    w_offset = board_w
    h_offset = board_h
    return canvas, new_coords, (w_ratio, h_ratio, w_offset, h_offset)


def corrds2dxdys(coordinates):
    # list of [x,y] --> [dx, dy, p1, p2, p3]
    # see paper 'A NEURAL REPRESENTATION OF SKETCH DRAWINGS' for details
    # BOS = [0, 0, 0, 0, 0]
    # EOS = [0, 0, 0, 0, 1]
    x0, y0 = 0, 0
    new_strokes = []
    for stroke in coordinates:
        for (x, y) in np.array(stroke).reshape((-1, 2)):
            p = np.array([x - x0, y - y0, 1, 0, 0])
            x0, y0 = x, y
            new_strokes.append(p)
        new_strokes[-1][2:] = [0, 1, 0]  # set the end of a stroke
    new_strokes.append([0, 0, 0, 0, 1])  # add EOS
    new_strokes = np.stack(new_strokes, axis=0)
    return new_strokes


def plot_dxdys(dxdys, split=True, w=64, h=64):
    xys = dxdys.copy()
    xys[:, 0] = np.cumsum(xys[:, 0])
    xys[:, 1] = np.cumsum(xys[:, 1])
    if split:
        ids = np.where(dxdys[:, 3] == 1)[0] + 1
        xys_split = np.split(xys, ids, axis=0)[:-1]  # split and remove the end pad tail
    else:
        xys_split = [xys]
    canvas = Image.new(mode='L', size=(w, h), color=255)
    draw = ImageDraw.Draw(canvas)
    for stroke in xys_split:
        xs, ys = stroke[:, 0], stroke[:, 1]
        xys = np.stack([xs, ys], axis=-1).reshape(-1)
        draw.line(xys.tolist(), fill=0, width=1)
    return canvas


import cv2


def plot_arrow(dxdys, split=True, w=256, h=256):
    xys = dxdys.copy()
    xys[:, 0] = np.cumsum(xys[:, 0]) / 64. * w
    xys[:, 1] = np.cumsum(xys[:, 1]) / 64. * h
    if split:
        ids = np.where(dxdys[:, 3] == 1)[0] + 1
        xys_split = np.split(xys, ids, axis=0)[:-1]  # split and remove the end pad tail
    else:
        xys_split = [xys]
    canvas = 255 * np.ones((h, w), dtype=np.uint8)
    for stroke in xys_split:
        xs, ys = stroke[:, 0], stroke[:, 1]
        xys = np.stack([xs, ys], axis=-1)
        xy0 = xys[0]
        for xy1 in xys[1:]:
            cv2.arrowedLine(canvas, tuple(xy0), tuple(xy1), 0, 1, 0, 0, 0.2)
            xy0 = xy1
    canvas = Image.fromarray(canvas)
    return canvas


import lmdb


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


class PotDataset(Dataset):
    def __init__(self, root=None, is_tarin=True, reduce=False):
        print('调试中，路径暂时指向内存')
        lmdb_path = os.path.join('/dev/shm/CASIA/data', os.path.basename(root))
        if not os.path.exists(lmdb_path):
            print('reading pot and generating lmdb cache file...')
            os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
            env = lmdb.open(lmdb_path, map_size=1099511627776)
            pots = sorted(glob.glob(os.path.join(root, '*.pot')))
            assert len(pots) > 0
            cnt = 0
            cache = {}
            for pot in tqdm.tqdm(pots):
                samples = read_pot(pot)
                for sample in samples:
                    data = {'tag_char': sample['tag_char'], 'coordinates': sample['coordinates']}
                    data_byte = pickle.dumps(data)
                    data_id = str(cnt).encode('utf-8')
                    cache[data_id] = data_byte
                    if cnt % 1000 == 0:
                        writeCache(env, cache)
                        cache = {}
                        print('Written %d' % (cnt))
                    cnt += 1
            cache['num_sample'.encode('utf-8')] = str(cnt).encode()
            writeCache(env, cache)
            print('save {} samples to {}'.format(cnt, lmdb_path))
        self.lmdb = lmdb.open(lmdb_path, max_readers=8, readonly=True, lock=False, readahead=False, meminit=False)
        self.is_train = is_tarin
        if is_tarin:
            self.img_h = cfg.TRAIN.IMG_H  # 64
            self.img_w = cfg.TRAIN.IMG_W
        else:
            self.img_h = cfg.TEST.IMG_H  # 64
            self.img_w = cfg.TEST.IMG_W
        grid_x, grid_y = np.meshgrid(np.arange(0, self.img_h), np.arange(0, self.img_w), indexing='ij')
        grid_x = grid_x / float(self.img_w)
        grid_y = grid_y / float(self.img_h)
        self.grid = np.stack((grid_x, grid_y), axis=0).astype(np.float32)
        self.max_len = -1  # 100 disable
        self.alphabet = ''  # '01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.cat_xy_grid = cfg.DATA_LOADER.CONCAT_GRID
        self.reduce = reduce
        with self.lmdb.begin(write=False) as txn:
            self.num_sample = int(txn.get('num_sample'.encode('utf-8')).decode())
            if len(self.alphabet) <= 0:
                self.indexes = list(range(0, self.num_sample))
            else:
                print('filter data out of alphabet')
                self.indexes = []
                for i in range(self.num_sample):
                    data_id = str(i).encode('utf-8')
                    data_byte = txn.get(data_id)
                    tag_char = pickle.loads(data_byte)['tag_char']
                    if tag_char in self.alphabet:
                        self.indexes.append(i)

    def __len__(self):
        if (not self.is_train) and self.reduce:
            return int(len(self.indexes) * 0.1)
        else:
            return len(self.indexes)

    def __getitem__(self, index):
        if self.is_train:
            index = index % (len(self))
        index = self.indexes[index]
        """渲染轨迹"""
        with self.lmdb.begin(write=False) as txn:
            data = pickle.loads(txn.get(str(index).encode('utf-8')))
            char_tag, coords = data['tag_char'], data['coordinates']
        if self.is_train and self.max_len > 0:
            l_seq = sum([len(l) // 2 for l in coords])
            if l_seq > self.max_len:
                print('skip {},{}'.format(index, char_tag))
                return self[index + 1]
        # print("len of coords is ", coords[0])
        thickness = random.randint(1, 3) if self.is_train else 2
        img_pil, coords_rend, _ = coords_render(coords, width=self.img_w, height=self.img_h, thickness=thickness)
        sk_pil, _, _ = coords_render(coords, width=self.img_w, height=self.img_h, thickness=1)
        sk_np = np.array(sk_pil)
        ys, xs = np.where(sk_np == 0)
        sk_points = np.stack((xs, ys), axis=-1).astype(np.float32)
        # 检查coord
        # plt.imshow(img_pil)
        # plt.show()
        # """按GMM模型预测值转换坐标"""
        label = corrds2dxdys(coords_rend)
        # plot_dxdys(label)
        # print(label)
        if self.cat_xy_grid:
            img_tensor = np.array(img_pil)
            img_tensor = np.concatenate(((np.expand_dims(img_tensor, axis=0) - 127.) / 127., self.grid), axis=0)
        else:
            img_tensor = (np.transpose(np.array(img_pil.convert('RGB')), (2, 0, 1)) - 127.) / 127.
        return {'img': torch.Tensor(img_tensor), 'label': torch.Tensor(label), 'sk_points': torch.Tensor(sk_points)}

    def collate_fn_(self, batch_data):
        bs = len(batch_data)
        max_len = max([s['label'].shape[0] for s in batch_data])
        EOS = torch.zeros(bs, max_len, 5)
        EOS[:, :, -1] = 1
        # EOS[:, :, [0, 1]] = 1000  # 把pad的(x,y)弄大一点，让dtw loss不要往这里对齐
        output = {'img': torch.zeros((bs, 3, self.img_h, self.img_w)),
                  'label': EOS, 'sk_points': [], 'o_scale': []}
        for i in range(bs):
            s = batch_data[i]['label'].shape[0]
            output['label'][i, :s] = batch_data[i]['label']
            output['img'][i] = batch_data[i]['img']
            output['sk_points'].append(batch_data[i]['sk_points'])
            output['o_scale'].append(batch_data[i]['o_scale'])
        return output

    def postprocess(self, dxdys, o_scale):
        """ 把预测坐标还原成CASIA格式 """
        """ 将预测轨迹还原成原图坐标"""
        xys = dxdys.copy()
        # dx,dy --> x,y
        xys[:, 0] = np.cumsum(xys[:, 0])
        xys[:, 1] = np.cumsum(xys[:, 1])
        # remove pad image pixel
        xys[:, 0] -= o_scale[0]
        xys[:, 1] -= o_scale[0]
        #  rescale x, y
        xys[:, 0] = xys[:, 0] / o_scale[1]
        xys[:, 1] = xys[:, 1] / o_scale[1]
        ids = np.where(dxdys[:, 3] == 1)[0] + 1
        xys_splits = np.split(xys, ids, axis=0)[:-1]  # split and remove the end pad tail
        res = []
        for splits in xys_splits:
            res.append(splits[:, [0, 1]].reshape(-1).tolist())
        return res


class resizeKeepRatio(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.board = 5
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        res_h = float(self.size[1] - self.board * 2)
        res_w = float(self.size[0] - self.board * 2)
        res = np.ones((int(res_h), int(res_w)), dtype=np.uint8) * 255
        # print('old',img.shape)
        if float(img.shape[0]) / img.shape[1] > float(res_h) / res_w:
            new_h = res_h  # 依高度缩放
            scale = new_h / img.shape[0]
            new_w = img.shape[1] * scale
            img = cv2.resize(img, (int(new_w), int(new_h)), interpolation=cv2.INTER_LINEAR)
        else:
            #  依宽度缩放
            new_w = res_w
            scale = new_w / img.shape[1]
            new_h = img.shape[0] * scale
            img = cv2.resize(img, (int(new_w), int(new_h)), interpolation=cv2.INTER_LINEAR)
        # print('new', (new_h, new_w))
        res[:int(new_h), :int(new_w)] = img
        res = np.pad(res, pad_width=((self.board, self.board), (self.board, self.board)),
                     mode='constant', constant_values=((255, 255), (255, 255)))
        # res = np.expand_dims(res, axis=0)
        # print('res', res.shape)
        return res, (self.board, scale)


class PotGntDataset(PotDataset):
    def __init__(self, root=None, is_tarin=True, reduce=False, interval=None):
        print('调试中，路径暂时指向copy lmdb')
        lmdb_path = os.path.join('data/data_lmdb',
                                 os.path.basename(root))
        if not os.path.exists(lmdb_path):
            print('reading pot and generating lmdb cache file...')
            os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
            env = lmdb.open(lmdb_path, map_size=1099511627776)
            pots = sorted(glob.glob(os.path.join(root, '*.pot')))
            gnts = sorted(glob.glob(os.path.join(root, '*.gnt')))
            pots_gnts = [[pots[i], gnts[i]] for i in range(min(len(pots), len(gnts)))]
            assert 0 < len(pots) == len(gnts)
            cnt = 0
            cache = {}
            for pot, gnt in tqdm.tqdm(pots_gnts):
                samples_pot = read_pot(pot)
                samples_gnt = read_gnt(gnt)
                samples_gnt, samples_pot = alignment_gnt_pot(samples_gnt, samples_pot)
                for i in range(len(samples_pot)):
                    sample_gnt = samples_gnt[i]
                    sample_pot = samples_pot[i]
                    data = {'tag_char': sample_pot['tag_char'], 'coordinates': sample_pot['coordinates'],
                            'tag_char_': sample_gnt['tag_char'], 'bitmap': sample_gnt['bitmap']}
                    data_byte = pickle.dumps(data)
                    data_id = str(cnt).encode('utf-8')
                    cache[data_id] = data_byte
                    if cnt % 1000 == 0:
                        writeCache(env, cache)
                        cache = {}
                        print('Written %d' % (cnt))
                    cnt += 1
            cache['num_sample'.encode('utf-8')] = str(cnt).encode()
            writeCache(env, cache)
            print('save {} samples to {}'.format(cnt, lmdb_path))
        self.lmdb = lmdb.open(lmdb_path, max_readers=8, readonly=True, lock=False, readahead=False, meminit=False)
        self.is_train = is_tarin
        if is_tarin:
            self.img_h = cfg.TRAIN.IMG_H  # 64
            self.img_w = cfg.TRAIN.IMG_W
        else:
            self.img_h = cfg.TEST.IMG_H  # 64
            self.img_w = cfg.TEST.IMG_W
        self.resize = resizeKeepRatio((self.img_w, self.img_h))
        grid_x, grid_y = np.meshgrid(np.arange(0, self.img_h), np.arange(0, self.img_w), indexing='ij')
        grid_x = grid_x / float(self.img_w)
        grid_y = grid_y / float(self.img_h)
        self.grid = np.stack((grid_x, grid_y), axis=0).astype(np.float32)
        self.max_len = -1  # 100 disable
        self.alphabet = ''  # '01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.cat_xy_grid = cfg.DATA_LOADER.CONCAT_GRID
        self.reduce = reduce
        if interval is not None:
            print('手动指定数据区间为[{},{})'.format(interval[0], interval[1]))
            self.indexes = range(interval[0], interval[1])
        else:
            with self.lmdb.begin(write=False) as txn:
                self.num_sample = int(txn.get('num_sample'.encode('utf-8')).decode())
                if len(self.alphabet) <= 0:
                    self.indexes = list(range(0, self.num_sample))
                else:
                    print('filter data out of alphabet')
                    self.indexes = []
                    for i in range(self.num_sample):
                        data_id = str(i).encode('utf-8')
                        data_byte = txn.get(data_id)
                        tag_char = pickle.loads(data_byte)['tag_char']
                        if tag_char in self.alphabet:
                            self.indexes.append(i)

    def __getitem__(self, index):
        if self.is_train:
            index = index % (len(self))
        index = self.indexes[index]
        """渲染轨迹"""
        with self.lmdb.begin(write=False) as txn:
            data = pickle.loads(txn.get(str(index).encode('utf-8')))
            tag_char, coords, bitmap = data['tag_char'], data['coordinates'], data['bitmap']
        if self.is_train and self.max_len > 0:
            l_seq = sum([len(l) // 2 for l in coords])
            if l_seq > self.max_len:
                print('skip {},{}'.format(index, tag_char))
                return self[index + 1]
        sk_pil, coords_rend, _ = coords_render(coords, width=self.img_w, height=self.img_h, thickness=1)
        img_arr, o_scale = self.resize(bitmap)
        img_pil = Image.fromarray(img_arr)
        # img_pil = Image.fromarray(bitmap).resize((self.img_w, self.img_h))
        # plt.imshow(sk_pil), plt.axis('off')
        # plt.savefig('debug/{}_sk.jpg'.format(index))
        # plt.imshow(img_pil)
        # plt.savefig('debug/{}_pil.jpg'.format(index))
        sk_np = np.array(sk_pil)
        ys, xs = np.where(sk_np == 0)
        sk_points = np.stack((xs, ys), axis=-1).astype(np.float32)
        # """按GMM模型预测值转换坐标"""
        label = corrds2dxdys(coords_rend) # (x, y) -> (dx, dy, down, up, end)
        if self.cat_xy_grid:  # 图片以及像素点坐标叠在一起
            img_tensor = np.array(img_pil)
            img_tensor = np.concatenate(((np.expand_dims(img_tensor, axis=0) - 127.) / 127., self.grid), axis=0)
        else:
            img_tensor = (np.transpose(np.array(img_pil.convert('RGB')), (2, 0, 1)) - 127.) / 127.
        return {'img': torch.Tensor(img_tensor), 'label': torch.Tensor(label),   # img: gnt读的图经过resize, label: pot读的坐标经过转换
                'sk_points': torch.Tensor(sk_points),               # sk_points: 根据pot的轨迹渲染出的图片的黑色点的坐标
                'o_scale': o_scale}     # 画布的空白边，gnt图像放缩比例（放缩后比放缩前）


def get_dataset_path(name):
    if name == 'train':
        path = 'data/data_raw/train'
    elif name == 'test':
        path = 'data/data_raw/test'
    elif name == 'debug':
        path = 'data/data_raw/debug'
    else:
        raise NotImplementedError
    return path


if __name__ == '__main__':
    dataset = PotGntDataset(root=get_dataset_path('debug'), is_tarin=True)
    dataset = PotGntDataset(root=get_dataset_path('train'), is_tarin=True)
    dataset = PotGntDataset(root=get_dataset_path('test'), is_tarin=True)
    # dataset_train = torch.utils.data.DataLoader(PotDataset(root=get_dataset_path('train'), is_tarin=True),
    #                                             batch_size=8, shuffle=False, drop_last=False, num_workers=0)
    # dataset_test = torch.utils.data.DataLoader(PotDataset(root=get_dataset_path('test'), is_tarin=True),
    #                                             batch_size=8, shuffle=False, drop_last=False, num_workers=0)
    # dataset = PotDataset(root=get_dataset_path('debug'), is_tarin=True)
    # dataset_debug = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn_,
    #                                            batch_size=8, shuffle=True, drop_last=False, num_workers=0)
    print('已生成数据缓存./cache')
    # for _ in tqdm.tqdm(dataset_train):
    #     pass
    # for _ in tqdm.tqdm(dataset_test):
    #     pass
    # for _ in tqdm.tqdm(datas`et_debug):
    #     pass
    # for _ in tqdm.tqdm(dataset_test):
    #     pass
    # train_loader_iter = iter(dataset_test)
    # datas = next(train_loader_iter)
    # print('len of datas is ',len(datas[0]))
