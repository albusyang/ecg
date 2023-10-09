import os
import pickle

def load(dirname):
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f, 'rb') as fid:  
        preproc = pickle.load(fid, encoding='latin1')  # 添加encoding参数以确保兼容性
    return preproc

def save(preproc, dirname):
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f, 'wb') as fid:  
        pickle.dump(preproc, fid, protocol=2)  # 使用protocol=2以确保兼容Python 2.7
