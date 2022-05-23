import tensorflow as tf
from tflearn import DNN
import time
import numpy as np
import argparse
import dlib
import cv2
import os
from skimage.feature import hog

from parameters import DATASET, TRAINING, NETWORK, VIDEO_PREDICTOR
from model import build_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

window_size = 24
window_step = 6


def load_model():
    with tf.Graph().as_default():
        print("===正在加载模型===")
        network = build_model()
        model = DNN(network)
        if os.path.isfile(TRAINING.save_model_path):
            model.load(TRAINING.save_model_path)
        else:
            print(">>> 文件路径 '{}' 出错了".format(TRAINING.save_model_path))
    return model

"""
首先通过人脸检测，检测出人脸框的位置，再使用加载的人脸关键点检测，进行人脸关键点的检测
"""
def get_landmarks(image, rects, predictor):
    if len(rects) > 1:
        print("TooManyFaces")
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        print("NoFaces")
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])



def predict(image, model, shape_predictor=None):
    if NETWORK.use_landmarks or NETWORK.use_hog_and_landmarks:
        face_rects = [dlib.rectangle(left=0, top=0, right=NETWORK.input_size, bottom=NETWORK.input_size)]
        face_landmarks = np.array([get_landmarks(image, face_rects, shape_predictor)])

        hog_features, _ = hog(image, orientations=9, pixels_per_cell=(16, 16),
                              cells_per_block=(1, 1), visualize=True)
        hog_features = np.asarray(hog_features)
        face_landmarks = face_landmarks.flatten()
        features = np.concatenate((face_landmarks, hog_features))

        tensor_image = image.reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
        predicted_label = model.predict([tensor_image, features.reshape((1, -1))])
        return get_emotion(predicted_label[0])
    else:
        tensor_image = image.reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
        predicted_label = model.predict(tensor_image)
        return get_emotion(predicted_label[0])
    return None


def get_emotion(label):
    if VIDEO_PREDICTOR.print_emotions:
        print("======")
        print("> 愤怒: {0:.1f}%\n> 厌恶: {1:.1f}%\n> 恐惧: {2:.1f}%"
            .format(label[0] * 100, label[1] * 100, label[2] * 100))
        print("> 快乐: {0:.1f}%\n> 悲伤: {1:.1f}%\n> 惊喜: {2:.1f}%"
              .format(label[3] * 100, label[4] * 100, label[5] * 100))
        print("> 中性: {0:.1f}%".format(label[6] * 100))
        print("======")
    label = label.tolist()
    return VIDEO_PREDICTOR.emotions[label.index(max(label))], max(label)



# model = load_model()
# image = cv2.imread("1.png", 0)  # 使用灰度图方式加载一张彩色照片
# # image = cv2.imread("2.png", 0)
# # image = cv2.imread("3503.jpg", 0)
# pred_img = cv2.resize(image, (NETWORK.input_size, NETWORK.input_size))
# image = np.array(pred_img)
# shape_predictor = dlib.shape_predictor(DATASET.shape_predictor_path)
# start_time = time.time()
# emotion, confidence = predict(image, model, shape_predictor)
# total_time = time.time() - start_time
# print("预测结果: {0} (自信把握: {1:.1f}%)".format(emotion, confidence * 100))
# print("总预测时间: {0:.1f} sec".format(total_time))

