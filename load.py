import json
import keras
import numpy as np
import os
import random
import scipy.io as sio
import tqdm

STEP = 256

def data_generator(batch_size, preproc, x, y):
    """生成器函数，用于产生批量的数据"""
    num_examples = len(x)
    examples = list(zip(x, y))  # 使用list将zip对象转为列表
    examples = sorted(examples, key=lambda x: x[0].shape[0])
    end = num_examples - batch_size + 1
    batches = [examples[i:i+batch_size]
               for i in range(0, end, batch_size)]
    random.shuffle(batches)
    while True:
        for batch in batches:
            x, y = zip(*batch)
            yield preproc.process(x, y)

class Preproc:
    """预处理类，用于处理ECG数据和标签"""

    def __init__(self, ecg, labels):
        self.mean, self.std = compute_mean_std(ecg)
        self.classes = sorted(set(l for label in labels for l in label))
        self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c: i for i, c in self.int_to_class.items()}

    def process(self, x, y):
        """处理x和y数据"""
        return self.process_x(x), self.process_y(y)

    def process_x(self, x):
        """处理x数据"""
        x = pad(x)
        x = (x - self.mean) / self.std
        x = x[:, :, None]
        return x

    def process_y(self, y):
        """处理y数据"""
        y = pad([[self.class_to_int[c] for c in s] for s in y], val=3, dtype=np.int32)
        y = keras.utils.np_utils.to_categorical(
            y, num_classes=len(self.classes))
        return y

def pad(x, val=0, dtype=np.float32):
    """填充函数，确保所有序列具有相同的长度"""
    max_len = max(len(i) for i in x)
    padded = np.full((len(x), max_len), val, dtype=dtype)
    for e, i in enumerate(x):
        padded[e, :len(i)] = i
    return padded

def compute_mean_std(x):
    """计算均值和标准差"""
    x = np.hstack(x)
    return (np.mean(x).astype(np.float32),
            np.std(x).astype(np.float32))

def load_dataset(data_json):
    """从JSON文件加载数据集"""
    with open(data_json, 'r') as fid:
        data = [json.loads(l) for l in fid]
    labels = []
    ecgs = []
    for d in tqdm.tqdm(data):
        labels.append(d['labels'])
        ecgs.append(load_ecg(d['ecg']))
    return ecgs, labels

def load_ecg(record):
    """加载ECG记录，支持.npy, .mat和其他格式"""
    if os.path.splitext(record)[1] == ".npy":
        ecg = np.load(record)
    elif os.path.splitext(record)[1] == ".mat":
        ecg = sio.loadmat(record)['val'].squeeze()
    else:  # Assumes binary 16 bit integers
        with open(record, 'rb') as fid:  # 注意这里的 'rb'
            ecg = np.fromfile(fid, dtype=np.int16)

    trunc_samp = STEP * int(len(ecg) / STEP)
    return ecg[:trunc_samp]

if __name__ == "__main__":
    data_json = "examples/cinc17/train.json"
    train = load_dataset(data_json)
    preproc = Preproc(*train)
    gen = data_generator(32, preproc, *train)
    for x, y in gen:
        print(x.shape, y.shape)
        break
