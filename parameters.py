import os


class Dataset:
    name = 'Fer2013'
    train_folder = 'fer2013_features/Training'  # 训练
    validation_folder = 'fer2013_features/PublicTest'  # 验证
    test_folder = 'fer2013_features/PrivateTest'  # 测试
    shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'  # 人脸关键点预测器
    num_d = 9000
    num_v = 100
    num_t = 20


class Network:
    model = 'B'
    input_size = 48
    output_size = 7
    activation = 'relu'  # 激活函数-relu
    loss = 'categorical_crossentropy'  # 损失函数-交叉熵损失函数
    use_landmarks = False
    use_hog_and_landmarks = True
    use_batchnorm_after_conv_layers = True  # 在卷积层之后使用批规范化（归一化）
    use_batchnorm_after_fully_connected_layers = True  # # 在全连接层之后使用批规范化


class Hyperparams:  # 超参数
    keep_prob = 0.956  # dropout = 1 - keep_prob 退出 = 1 - 保持概率
    learning_rate = 0.016  # 学习率

    learning_rate_decay = 0.864  # 衰减-学习率递减
    decay_step = 50  # 衰减步长-用来控制衰减速度

    optimizer = 'momentum'  # 常用优化方法（optimizer-优化器）
    optimizer_param = 0.95  # 优化参数


class Training:
    batch_size = 128  # 批次大小
    epochs = 13  # 迭代次数
    vizualize = True  # 可视化
    snapshot_step = 500  # snapshot_step参数里面设置隔几步保存一次模型
    logs_dir = "logs"
    checkpoint_dir = "checkpoints/chk"
    best_checkpoint_path = "checkpoints/best/"
    max_checkpoints = 1
    checkpoint_frequency = 1.0  # 1小时
    save_model = True
    save_model_path = "best_model/saved_model.bin"


class VideoPredictor:
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    print_emotions = True
    # 人脸识别分类检测器
    face_detection_classifier = "lbpcascade_frontalface.xml"
    show_confidence = True


def make_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


DATASET = Dataset()
NETWORK = Network()
TRAINING = Training()
HYPERPARAMS = Hyperparams()
VIDEO_PREDICTOR = VideoPredictor()

make_dir(TRAINING.logs_dir)
make_dir(TRAINING.checkpoint_dir)
