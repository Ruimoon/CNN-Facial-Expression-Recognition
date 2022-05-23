from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.estimator import regression
from tflearn.optimizers import Momentum, Adam

from parameters import NETWORK, HYPERPARAMS


def build_model(optimizer=HYPERPARAMS.optimizer, optimizer_param=HYPERPARAMS.optimizer_param,
                learning_rate=HYPERPARAMS.learning_rate, keep_prob=HYPERPARAMS.keep_prob,
                learning_rate_decay=HYPERPARAMS.learning_rate_decay, decay_step=HYPERPARAMS.decay_step):
    if NETWORK.model == 'B':
        return build_modelB(optimizer, optimizer_param, learning_rate, keep_prob, learning_rate_decay, decay_step)
    else:
        print("错误模型" + str(NETWORK.model))
        exit()



def build_modelB(optimizer=HYPERPARAMS.optimizer, optimizer_param=HYPERPARAMS.optimizer_param,
                 learning_rate=HYPERPARAMS.learning_rate, keep_prob=HYPERPARAMS.keep_prob,
                 learning_rate_decay=HYPERPARAMS.learning_rate_decay, decay_step=HYPERPARAMS.decay_step):
    images_network = input_data(shape=[None, NETWORK.input_size, NETWORK.input_size, 1], name='input_1')

    images_network = conv_2d(images_network, 64, 3, activation=NETWORK.activation)
    # images_network = local_response_normalization(images_network)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = max_pool_2d(images_network, 3, strides=2)

    images_network = conv_2d(images_network, 128, 3, activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = max_pool_2d(images_network, 3, strides=2)

    images_network = conv_2d(images_network, 256, 3, activation=NETWORK.activation)
    if NETWORK.use_batchnorm_after_conv_layers:
        images_network = batch_normalization(images_network)
    images_network = max_pool_2d(images_network, 3, strides=2)

    images_network = dropout(images_network, keep_prob=keep_prob)
    images_network = fully_connected(images_network, 4096, activation=NETWORK.activation)

    images_network = dropout(images_network, keep_prob=keep_prob)
    images_network = fully_connected(images_network, 1024, activation=NETWORK.activation)

    if NETWORK.use_batchnorm_after_fully_connected_layers:
        images_network = batch_normalization(images_network)

    if NETWORK.use_landmarks or NETWORK.use_hog_and_landmarks:
        if NETWORK.use_hog_and_landmarks:
            landmarks_network = input_data(shape=[None, 217], name='input_2')
        else:
            landmarks_network = input_data(shape=[None, 68, 2], name='input_2')

        landmarks_network = fully_connected(landmarks_network, 1024, activation=NETWORK.activation)

        if NETWORK.use_batchnorm_after_fully_connected_layers:
            landmarks_network = batch_normalization(landmarks_network)

        landmarks_network = fully_connected(landmarks_network, 128, activation=NETWORK.activation)

        if NETWORK.use_batchnorm_after_fully_connected_layers:
            landmarks_network = batch_normalization(landmarks_network)

        images_network = fully_connected(images_network, 128, activation=NETWORK.activation)

        # merge函数是将tensor列表合并成一个
        network = merge([images_network, landmarks_network], 'concat', axis=1)
    else:
        network = images_network

    network = fully_connected(network, NETWORK.output_size, activation='softmax')

    if optimizer == 'momentum':
        optimizer = Momentum(learning_rate=learning_rate, momentum=optimizer_param,
                             lr_decay=learning_rate_decay, decay_step=decay_step)
    else:
        print("无效"+optimizer)
    network = regression(network, optimizer=optimizer, loss=NETWORK.loss, learning_rate=learning_rate, name='out_put')

    return network


