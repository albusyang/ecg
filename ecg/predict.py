import argparse
import numpy as np
import tensorflow as tf
import os

import load
import util

def predict(data_json, model_path):
    """预测函数，加载模型并对数据进行预测"""
    # 加载预处理对象
    preproc = util.load(os.path.dirname(model_path))
    # 加载数据集
    dataset = load.load_dataset(data_json)
    # 对数据进行预处理
    x, y = preproc.process(*dataset)
    # 加载模型
    model = tf.keras.models.load_model(model_path)
    # 进行预测
    probs = model.predict(x, verbose=1)

    return probs

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加数据json文件路径参数
    parser.add_argument("data_json", help="path to data json")
    # 添加模型路径参数
    parser.add_argument("model_path", help="path to model")
    # 解析命令行参数
    args = parser.parse_args()
    # 使用模型进行预测
    probs = predict(args.data_json, args.model_path)
