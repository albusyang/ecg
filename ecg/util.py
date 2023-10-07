import os
import pickle

def load(dirname):
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f, 'rb') as fid:  # 注意这里的 'rb'
        preproc = pickle.load(fid)
    return preproc

def save(preproc, dirname):
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f, 'wb') as fid:  # 注意这里的 'wb'
        pickle.dump(preproc, fid)
