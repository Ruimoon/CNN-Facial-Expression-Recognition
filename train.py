import tensorflow as tf
from tflearn import DNN
import time
import os

from data_loader import load_data
from parameters import DATASET, TRAINING, HYPERPARAMS, NETWORK
from model import build_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train(optimizer=HYPERPARAMS.optimizer, optimizer_param=HYPERPARAMS.optimizer_param,
          learning_rate=HYPERPARAMS.learning_rate, keep_prob=HYPERPARAMS.keep_prob,
          learning_rate_decay=HYPERPARAMS.learning_rate_decay, decay_step=HYPERPARAMS.decay_step,
          train_model=False):
    print("===加载数据集" + DATASET.name + "===")
    if train_model:
        data, validation = load_data(validation=True)
    else:
        data, validation, test = load_data(validation=True, test=True)

    with tf.Graph().as_default():
        print("===创建CNN模型中===")
        network = build_model(optimizer, optimizer_param, learning_rate,
                              keep_prob, learning_rate_decay, decay_step)
        model = DNN(network, tensorboard_dir=TRAINING.logs_dir,
                    tensorboard_verbose=0, checkpoint_path=TRAINING.checkpoint_dir,
                    max_checkpoints=TRAINING.max_checkpoints)

        # tflearn.config.init_graph(seed=None, log_device=False, num_cores=6)

        if train_model:
            # Training phase
            print("===开始训练===")
            print(">>> optimizer = '{}'".format(optimizer))
            print(">>> learning_rate = {}".format(learning_rate))
            print(">>> learning_rate_decay = {}".format(learning_rate_decay))
            print(">>> keep_prob = {}".format(keep_prob))
            print(">>> epochs = {}".format(TRAINING.epochs))
            print(">>> use landmarks = {}".format(NETWORK.use_landmarks))
            print(">>> use hog + landmarks = {}".format(NETWORK.use_hog_and_landmarks))

            start_time = time.time()
            if NETWORK.use_landmarks or NETWORK.use_hog_and_landmarks:
                model.fit([data['X'], data['X2']], data['Y'],
                          validation_set=([validation['X'], validation['X2']], validation['Y']),
                          snapshot_step=TRAINING.snapshot_step,
                          show_metric=TRAINING.vizualize,
                          batch_size=TRAINING.batch_size,
                          n_epoch=TRAINING.epochs)
            else:
                model.fit(data['X'], data['Y'],
                          validation_set=(validation['X'], validation['Y']),
                          snapshot_step=TRAINING.snapshot_step,
                          show_metric=TRAINING.vizualize,
                          batch_size=TRAINING.batch_size,
                          n_epoch=TRAINING.epochs)
                validation['X2'] = None
            training_time = time.time() - start_time
            print("训练时间总共为 {0:.1f} sec".format(training_time))

            if TRAINING.save_model:
                print("===保存模型===")
                model.save(TRAINING.save_model_path)
                if not (os.path.isfile(TRAINING.save_model_path)) and \
                        os.path.isfile(TRAINING.save_model_path + ".meta"):
                    os.rename(TRAINING.save_model_path + ".meta", TRAINING.save_model_path)
            print("===开始评估===")
            validation_accuracy = evaluate(model, validation['X'], validation['X2'], validation['Y'])
            print(">>> 评估准确率 = {0:.1f}".format(validation_accuracy * 100))
            return validation_accuracy

        else:
            print("===开始评估===")
            print("===加载模型中===")
            if os.path.isfile(TRAINING.save_model_path):
                model.load(TRAINING.save_model_path)
            else:
                print("文件 '{}' 未能找到".format(TRAINING.save_model_path))
                exit()

            if not (NETWORK.use_landmarks or NETWORK.use_hog_and_landmarks):
                validation['X2'] = None
                test['X2'] = None

            print(">>> 评估样本: {}".format(len(validation['Y'])))
            print(">>> 测试样本: {}".format(len(test['Y'])))
            print("===评估中===")
            start_time = time.time()
            validation_accuracy = evaluate(model, validation['X'], validation['X2'], validation['Y'])
            print(">>> 评估准确率 = {0:.1f}".format(validation_accuracy * 100))
            test_accuracy = evaluate(model, test['X'], test['X2'], test['Y'])
            print(">>> 测试准确率 = {0:.1f}".format(test_accuracy * 100))
            print(">>> 评估时间为 = {0:.1f} sec".format(time.time() - start_time))
            return test_accuracy


def evaluate(model, X, X2, Y):
    if NETWORK.use_landmarks or NETWORK.use_hog_and_landmarks:
        accuracy = model.evaluate([X, X2], Y)
    else:
        accuracy = model.evaluate(X, Y)
    return accuracy[0]

train()
# tensorboard --logdir=logs