from parameters import DATASET, NETWORK
import numpy as np


def load_data(validation=False, test=False):
    data_dict = dict()
    validation_dict = dict()
    test_dict = dict()

    if DATASET.name == "Fer2013":
        # 加载训练集
        data_dict['X'] = np.load(DATASET.train_folder + '/images.npy')
        data_dict['X'] = data_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
        if NETWORK.use_landmarks:
            data_dict['X2'] = np.load(DATASET.train_folder + '/landmarks.npy')
        if NETWORK.use_hog_and_landmarks:
            data_dict['X2'] = np.load(DATASET.train_folder + '/landmarks.npy')
            data_dict['X2'] = np.array([x.flatten() for x in data_dict['X2']])
            # (28709, 136)
            # x.flatten()把数组降到一位 因为是68x2
            data_dict['X2'] = np.concatenate((data_dict['X2'],
                                              np.load(DATASET.train_folder + '/hog_features.npy')),
                                             axis=1)
        # 照第二维度进行拼接得到 [,217]
        data_dict['Y'] = np.load(DATASET.train_folder + '/labels.npy')

        data_dict['X'] = data_dict['X'][0:DATASET.num_d, :, :]
        if NETWORK.use_hog_and_landmarks:
            data_dict['X2'] = data_dict['X2'][0:DATASET.num_d, :]
            #     [DATASET.num_d,217]
        elif NETWORK.use_landmarks:
            data_dict['X2'] = data_dict['X2'][0:DATASET.num_d, :, :]
        #     [DATASET.num_d,68,2]
        data_dict['Y'] = data_dict['Y'][0:DATASET.num_d, :]

        if validation:
            validation_dict['X'] = np.load(DATASET.validation_folder + '/images.npy')
            validation_dict['X'] = validation_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
            if NETWORK.use_landmarks:
                validation_dict['X2'] = np.load(DATASET.validation_folder + '/landmarks.npy')
            if NETWORK.use_hog_and_landmarks:
                validation_dict['X2'] = np.load(DATASET.validation_folder + '/landmarks.npy')
                validation_dict['X2'] = np.array([x.flatten() for x in validation_dict['X2']])
                validation_dict['X2'] = np.concatenate(
                    (validation_dict['X2'], np.load(DATASET.validation_folder + '/hog_features.npy')), axis=1)
            validation_dict['Y'] = np.load(DATASET.validation_folder + '/labels.npy')

            validation_dict['X'] = validation_dict['X'][0:DATASET.num_v, :, :]
            if NETWORK.use_hog_and_landmarks:
                validation_dict['X2'] = validation_dict['X2'][0:DATASET.num_v, :]
            elif NETWORK.use_landmarks:
                validation_dict['X2'] = validation_dict['X2'][0:DATASET.num_v, :, :]
            validation_dict['Y'] = validation_dict['Y'][0:DATASET.num_v, :]

        if test:
            test_dict['X'] = np.load(DATASET.test_folder + '/images.npy')
            test_dict['X'] = test_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
            if NETWORK.use_landmarks:
                test_dict['X2'] = np.load(DATASET.test_folder + '/landmarks.npy')
            if NETWORK.use_hog_and_landmarks:
                test_dict['X2'] = np.load(DATASET.test_folder + '/landmarks.npy')
                test_dict['X2'] = np.array([x.flatten() for x in test_dict['X2']])
                test_dict['X2'] = np.concatenate((test_dict['X2'], np.load(DATASET.test_folder + '/hog_features.npy')),
                                                 axis=1)
            test_dict['Y'] = np.load(DATASET.test_folder + '/labels.npy')

            test_dict['X'] = test_dict['X'][0:DATASET.num_t, :, :]
            if NETWORK.use_hog_and_landmarks:
                test_dict['X2'] = test_dict['X2'][0:DATASET.num_t, :]
            elif NETWORK.use_landmarks:
                test_dict['X2'] = test_dict['X2'][0:DATASET.num_t, :, :]
            test_dict['Y'] = test_dict['Y'][0:DATASET.num_t, :]

        if not validation and not test:
            return data_dict
        elif not test:
            return data_dict, validation_dict
        else:
            return data_dict, validation_dict, test_dict
    else:
        print("未知数据集")
        exit()

