import json
import keras
import numpy as np
import scipy.io as sio
import scipy.stats as sst

import load
import network
import util

def predict(record):
    """预测给定记录的类别"""
    ecg = load.load_ecg(record + ".mat")
    preproc = util.load(".")
    x = preproc.process_x([ecg])

    with open("config.json", 'r') as f:  # 使用with语句打开文件
        params = json.load(f)
    params.update({
        "compile": False,
        "input_shape": [None, 1],
        "num_categories": len(preproc.classes)
    })

    model = network.build_network(**params)
    model.load_weights('model.hdf5')

    probs = model.predict(x)
    prediction = sst.mode(np.argmax(probs, axis=2).squeeze())[0][0]
    return preproc.int_to_class[prediction]

if __name__ == '__main__':
    import sys
    print(predict(sys.argv[1]))  # 使用print函数的括号语法
