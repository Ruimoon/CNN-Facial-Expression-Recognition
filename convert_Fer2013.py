import numpy as np
import pandas as pd
import os
import errno
import scipy.misc
import time
import dlib
import cv2

from skimage.feature import hog

# 初始化
image_height = 48
image_width = 48
window_size = 24
window_step = 6
ONE_HOT_ENCODING = True
SAVE_IMAGES = True
GET_LANDMARKS = True
GET_HOG_FEATURES = True
GET_HOG_IMAGES = True
GET_HOG_WINDOWS_FEATURES = True
SELECTED_LABELS = [0, 1, 2, 3, 4, 5, 6]
OUTPUT_FOLDER_NAME = "fer2013_features"


# 加载 Dlib 预测器并准备数组
print("===开始准备===")
# 使用模型构建特征提取器 功能：标记人脸关键点
# 参数：‘data/data_dlib/shape_predictor_68_face_landmarks.dat’：68个关键点模型地址
# 返回值：人脸关键点预测器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

new_labels = [0, 1, 2, 3, 4, 5, 6]


try:
    os.makedirs("fer2013_features")
except OSError as e:
    if e.errno == errno.EEXIST and os.path.isdir("fer2013_features"):  # 判断某一路径是否为目录
        pass
    else:
        raise

"""

"""
# get_landmarks()函数将一个图像转化成numpy数组，并返回一个68×2元素矩阵，输入图像的每个特征点对应每行的一个x，y坐标。
# 特征提取器（predictor）需要一个粗糙的边界框作为算法输入，由一个传统的能返回一个矩形列表的人脸检测器（detector）提供，其每个矩形列表在图像中对应一个脸。
def get_landmarks(image, rects):  # rects = [48x48]
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])


# 一位有效编码
# 这样做的好处主要有：
#   解决了分类器不好处理属性数据的问题
#   在一定程度上也起到了扩充特征的作用
"""
true：返回列表型表情标签一位编码
false：返回表情标签，也就是索引位置
"""
def get_new_label(label, one_hot_encoding=False):
    if one_hot_encoding:
        new_label = new_labels.index(label)  # 从列表中找出某个值第一个匹配项的索引位置 eg:label = 1  ----> 1
        label = list(np.zeros(len(new_labels), 'uint8'))  # [0,0,0,0,0,0,0]
        label[new_label] = 1  # [0,1,0,0,0,0,0] label位为1
        return label  # [0,1,0,0,0,0,0]
    else:
        return new_labels.index(label)  # 得到索引位置


print("===导入csv文件===")
data = pd.read_csv('fer2013.csv')
start_time = time.time()
for category in data['Usage'].unique():  # 类别：Training, PublicTest, PrivateTest
    print("转换数据集: " + category + "-->")  # 转换集
    # create folder
    if not os.path.exists(category):
        try:
            os.makedirs("fer2013_features" + '/' + category)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir("fer2013_features"):
                pass
            else:
                raise

    # 获取实际类别的样本和标签
    category_data = data[data['Usage'] == category]  # 单独得到对应的类别的所有数据：emotion-pixels-Usage
    samples = category_data['pixels'].values  # 对应类别的图片数组-所有像素点<class 'numpy.ndarray'>
    labels = category_data['emotion'].values  # 对应类别的所有表情编号

    # 获取图像并提取特征
    images = []
    labels_list = []
    landmarks = []
    hog_features = []
    hog_images = []
    nb_images_per_label = [0, 0, 0, 0, 0, 0, 0]
    for i in range(len(samples)):  # 32298+3589+3589=35887
        try:
            if labels[i] in SELECTED_LABELS:
                image = np.fromstring(samples[i], dtype=int, sep=" ").reshape((image_height, image_width))
                # image = image.astype(np.uint8)
                # samples[i]->str  np.fromstring可以很方便的从字符串中进行解码出数据。
                #               这个对于数据传输和信息解析非常的方便。创建一个新的一维数组，该数组从字符串中的文本数据初始化。
                # 48行x48列 一共2304个像素点
                images.append(image)  # 存放对应类别的图片，分为48x48像素二维数组的列表

                if SAVE_IMAGES:
                    scipy.misc.imsave("fer2013_features" + '/' + category + '/' + str(i) + '.jpg', image)
                    # 用于储存图片，将数组保存为图像。imageio.imwrite()
                    # imageio.imsave( + '/' + category + '/' + str(i) + '.jpg', image)

                if GET_HOG_FEATURES:
                    features, hog_image = hog(image, orientations=9, pixels_per_cell=(16, 16),
                                              cells_per_block=(1, 1), visualize=True)
                    # image：可以是灰度图或者彩色图 (M，N [，C]) ndarray 输入图像
                    # orientations：int，直方图bin的数量。就是把180度分成几份，也就是bin的数量；
                    #               注意角度的范围介于0到180度之间
                    #               ，而不是0到360度， 这被称为“无符号”梯度，
                    #                因为两个完全相反的方向被认为是相同的。
                    # pixels_per_cell：一个cell里包含的像素个数；2元组(int，int)，可选的单元格大小（以像素为单位）。
                    # cells_per_block：一个block包含的cell个数；2元组(int，int)，可选每个块中的单元格数。
                    # visualize：是否返回一个hog图像用于显示，下面会显示这张图；
                    #           bool，可选,同时返回HOG的图像。对于每个单元格和方向箱，
                    #           图像包含以单元格中心为中心的线段，垂直于所跨越的角度范围的中点方向箱，
                    #           并具有与相应的强度成比例的直方图值。
                    # 存入hog特征
                    hog_features.append(features)
                    if GET_HOG_IMAGES:
                        # 存入hog图片
                        hog_images.append(hog_image)

                if GET_LANDMARKS:
                    # imageio.imsave('temp.jpg', image)
                    scipy.misc.imsave('temp.jpg', image)
                    image2 = cv2.imread('temp.jpg')
                    face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
                    # rect.left(), rect.top(), rect.right(), rect.bottom()
                    face_landmarks = get_landmarks(image2, face_rects)
                    # 存入landmark特征
                    landmarks.append(face_landmarks)

                # 存入表情每个标签
                labels_list.append(get_new_label(labels[i], one_hot_encoding=ONE_HOT_ENCODING))
                nb_images_per_label[get_new_label(labels[i])] += 1  # 人脸表情标签个数加1，最终得到每个表情的个数

        except Exception as e:
            print("数据处理错误: 图片" + str(i) + " - " + str(e))
    print("===对应处理数据保存完毕===")
    print("表情数：")
    print(nb_images_per_label)

    print("===开始保存数据===")
    np.save("fer2013_features" + '/' + category + '/images.npy', images)
    if ONE_HOT_ENCODING:
        np.save("fer2013_features" + '/' + category + '/labels.npy', labels_list)
        print("<one-hot-labels>编码保存完毕")
    else:
        np.save("fer2013_features" + '/' + category + '/labels.npy', labels_list)
        print("<labels>保存完毕")
    if GET_LANDMARKS:
        np.save("fer2013_features" + '/' + category + '/landmarks.npy', landmarks)
        print("<landmarks>保存完毕")
    if GET_HOG_FEATURES or GET_HOG_WINDOWS_FEATURES:
        np.save("fer2013_features" + '/' + category + '/hog_features.npy', hog_features)
        print("<hog_features>保存完毕")
        if GET_HOG_IMAGES:
            np.save("fer2013_features" + '/' + category + '/hog_images.npy', hog_images)
            print("<hog_images>保存完毕")
training_time = time.time() - start_time
print("训练时间总共为 {0:.1f} sec".format(training_time))